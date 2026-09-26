"""
perfis.py
=========
Constrói os dois tipos de perfis estilométricos do artigo:
  1) **Trabalho único (Baseline)**: 1 único texto por estudante (submissão aleatória ou primeira)
  2) **Perfil longitudinal**: agregado de N submissões (média ou mediana) do mesmo estudante

Conforme §3.3 e §4.2 do Artigo V2:
  - Longitudinal usa TODAS as submissões (média 11 textos/estudante no artigo real)
  - Baseline usa 1 submissão escolhida como "texto de referência"
  - Para cada par (perfil, texto questionado) calcula-se vetor de diferenças ou distâncias
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from .config import SEED

rng = np.random.default_rng(SEED)


def dividir_por_estudante(df_features: pd.DataFrame,
                          col_estudante: str = 'estudante') -> Dict[str, pd.DataFrame]:
    """Agrupa DataFrame de features por estudante → Dict[id_estudante] = DataFrame submissões."""
    return {est: sub.copy().reset_index(drop=True) for est, sub in df_features.groupby(col_estudante)}


def construir_perfil_longitudinal(submissoes_estudante: pd.DataFrame,
                                  lista_features: List[str],
                                  metodo_agregacao: str = 'mean') -> pd.Series:
    """
    Agrega múltiplas submissões do mesmo estudante → vetor de perfil (1 valor por feature).
    :param submissoes_estudante: DataFrame submissões (1 linha por submissão)
    :param lista_features: nomes das colunas de features
    :param metodo_agregacao: 'mean' (padrão), 'median', 'trimmed_mean'
    :return: pd.Series com médias por feature
    """
    vals = submissoes_estudante[lista_features].values.astype(float)
    if metodo_agregacao == 'median':
        perfil = np.nanmedian(vals, axis=0)
    elif metodo_agregacao == 'trimmed_mean':
        from scipy.stats import trim_mean
        perfil = trim_mean(vals, proportiontocut=0.1, axis=0)
    else:  # mean
        perfil = np.nanmean(vals, axis=0)
    perfil = np.where(np.isfinite(perfil), perfil, 0.0)
    return pd.Series(perfil, index=lista_features)


def construir_perfil_baseline(submissoes_estudante: pd.DataFrame,
                              lista_features: List[str],
                              modo_escolha: str = 'aleatoria',
                              idx_fixo: Optional[int] = None) -> pd.Series:
    """
    Retorna 1 submissão como perfil de baseline.
    :param modo_escolha: 'primeira', 'ultima', 'aleatoria', 'fixa' (idx_fixo)
    """
    n_sub = len(submissoes_estudante)
    if idx_fixo is not None and 0 <= idx_fixo < n_sub:
        idx = idx_fixo
    elif modo_escolha == 'primeira':
        idx = 0
    elif modo_escolha == 'ultima':
        idx = n_sub - 1
    else:  # aleatoria
        idx = rng.integers(0, n_sub)
    return submissoes_estudante[lista_features].iloc[idx].astype(float)


# ====================================================================
# Construção de pares (perfil, texto questionado) → matriz de dataset para classificação SVM
# ====================================================================
def _vetor_diferencas(perfil: pd.Series, texto_questionado: pd.Series, lista_features: List[str]) -> np.ndarray:
    a = perfil[lista_features].values.astype(float)
    b = texto_questionado[lista_features].values.astype(float)
    # Vetor de diferenças normalizadas z (padronizado)
    return np.abs(a - b)


def gerar_pares_classificacao(
    df_features: pd.DataFrame,
    lista_features: List[str],
    col_estudante: str = 'estudante',
    col_id_texto: str = 'id_texto',
    col_autoria: str = 'autoria',  # 'humana','hibrida','ia_assistida','ia_gerada'
    modo_agreg_long: str = 'mean',
    modo_baseline: str = 'primeira',
    usar_dataset_longitudinal: bool = True,
    seed_gerar: int = SEED,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gera DataFrame com pares (perfil X texto questionado) para treinar o SVM.
    Para CADA estudante:
      - Constroi perfil (longitudinal ou baseline)
      - Para CADA submissão do estudante: gera par (perfil, submissão) → classe 0 (autêntica/humana)
      - Para CADA submissão de OUTROS estudantes + textos IA (se existirem): classe 1 (não autêntica)

    Retorna:
      - df_pares: colunas 'estudante','id_texto_perfil','id_texto_test','autoria_test','tipo_perfil',
                  colunas 'diff_{feature}' (204 colunas), 'y' (0=autêntico, 1=não-autêntico)
      - cols_diff: nomes das colunas de diferenças
    """
    rng = np.random.default_rng(seed_gerar)
    por_estudante = dividir_por_estudante(df_features, col_estudante)
    cols_diff = [f'diff_{f}' for f in lista_features]
    linhas = []

    tipo_perfil_str = 'longitudinal' if usar_dataset_longitudinal else 'baseline'

    for estudante_alvo, df_subs in por_estudante.items():
        # 1) construir perfil do estudante_alvo
        if usar_dataset_longitudinal:
            perfil = construir_perfil_longitudinal(df_subs, lista_features, metodo_agregacao=modo_agreg_long)
        else:
            # baseline: remover uma submissão para ser o perfil (1 trabalho), as restantes são textos testados
            if len(df_subs) <= 1:
                continue
            sub_idx_perfil = 0 if modo_baseline == 'primeira' else int(rng.integers(0, len(df_subs)))
            perfil = df_subs[lista_features].iloc[sub_idx_perfil].astype(float)
            # excluir a própria submissão de perfil dos testes
            df_tests_mesmo_estudante = df_subs.drop(df_subs.index[sub_idx_perfil]).copy()
        if usar_dataset_longitudinal:
            df_tests_mesmo_estudante = df_subs.copy()

        # 2) pares classe 0 (autênticos → mesmos textos do estudante_alvo)
        id_perfil = f'{estudante_alvo}_perfil_{tipo_perfil_str}'
        linhas_classe0_est = []
        for _, linha_test in df_tests_mesmo_estudante.iterrows():
            diff = _vetor_diferencas(perfil, linha_test, lista_features)
            linhas_classe0_est.append({
                'estudante': estudante_alvo,
                'id_texto_perfil': id_perfil,
                'id_texto_test': linha_test.get(col_id_texto, ''),
                'autoria_test': linha_test.get(col_autoria, 'humana'),
                'tipo_perfil': tipo_perfil_str,
                **{c: v for c, v in zip(cols_diff, diff)},
                'y': 0,
            })
        linhas.extend(linhas_classe0_est)

        # 3) pares classe 1 (não autênticos → textos dos OUTROS estudantes, ou IA, ou híbridos)
        #    — colecionar primeiro TODOS os textos de outros, depois undersamplear para n_classe0_por_estudante
        n_classe0 = len(linhas_classe0_est)
        todos_outros = []
        for outro_est, df_outro_subs in por_estudante.items():
            if outro_est == estudante_alvo:
                continue
            for _, linha_test in df_outro_subs.iterrows():
                todos_outros.append(linha_test)
        # Selecionar exatamente n_classe0 exemplos de classe 1 para este estudante (mantém 50/50)
        n_classe1 = min(n_classe0, len(todos_outros))
        if n_classe1 > 0 and len(todos_outros) > 0:
            # amostra aleatória, reproduzível
            idx_amostra = rng.choice(len(todos_outros), size=n_classe1, replace=False)
            for idx in idx_amostra:
                linha_test = todos_outros[idx]
                diff = _vetor_diferencas(perfil, linha_test, lista_features)
                linhas.append({
                    'estudante': estudante_alvo,
                    'id_texto_perfil': id_perfil,
                    'id_texto_test': linha_test.get(col_id_texto, ''),
                    'autoria_test': linha_test.get(col_autoria, 'outro_estudante'),
                    'tipo_perfil': tipo_perfil_str,
                    **{c: v for c, v in zip(cols_diff, diff)},
                    'y': 1,
                })

    df_pares = pd.DataFrame(linhas).reset_index(drop=True)

    # GARANTIR BALANCEAMENTO FINAL (undersample da classe maioritária para 50/50)
    if len(df_pares) > 0 and df_pares['y'].nunique() >= 2:
        n0 = (df_pares['y'] == 0).sum()
        n1 = (df_pares['y'] == 1).sum()
        min_n = min(n0, n1)
        if n0 > min_n:
            df_pares = pd.concat([
                df_pares[df_pares['y'] == 0].sample(n=min_n, random_state=seed_gerar),
                df_pares[df_pares['y'] == 1],
            ], ignore_index=True).sample(frac=1, random_state=seed_gerar).reset_index(drop=True)
        elif n1 > min_n:
            df_pares = pd.concat([
                df_pares[df_pares['y'] == 0],
                df_pares[df_pares['y'] == 1].sample(n=min_n, random_state=seed_gerar),
            ], ignore_index=True).sample(frac=1, random_state=seed_gerar).reset_index(drop=True)

    return df_pares, cols_diff
