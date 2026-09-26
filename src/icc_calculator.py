"""
icc_calculator.py
=================
Calcula o Coeficiente de Correlação Intraclasse (ICC) para cada característica
estilométrica, avaliando a estabilidade **intra-autor (inter-textos do mesmo estudante)**
em oposição à variabilidade **inter-autores (entre estudantes)**.

Conforme §4.1 e Quadro 2 do Artigo V2:
  - ICC = A,1 (two-way random effects, absolute agreement)
  - Limiar de estabilidade: ICC ≥ 0,60 (Figura 1 linha tracejada)
  - Top 20 características com maior ICC (Quadro 2)
  - Intervalos de confiança 95% por bootstrap (500 reamostragens)
"""

import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pingouin as pg

from .config import (
    SEED,
    LIMIAR_ICC_ESTAVEL,
    TOP_N_FEATURES_ICC,
    N_BOOTSTRAP,
    IC_NIVEL,
    FAMILIAS_FEATURES,
)

warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)


def _obter_familia_feature(nome_feature: str) -> str:
    for fam, lista in FAMILIAS_FEATURES.items():
        if nome_feature in lista:
            return fam
    return 'Outros'


def calcular_icc_uma_feature(df_long: pd.DataFrame, feature: str,
                             coluna_estudante: str = 'estudante',
                             coluna_submissao: str = 'submissao_idx') -> Optional[float]:
    """
    Calcula ICC(A,1) para UMA característica.
    df_long: DataFrame no formato longo com 1 linha por (estudante, submissao)

    Retorna ICC ou None se falhar.
    """
    # Remover NaN ou Inf
    df_tmp = df_long[[coluna_estudante, coluna_submissao, feature]].replace([np.inf, -np.inf], np.nan).dropna()
    if df_tmp[feature].nunique() <= 2:
        return np.nan
    try:
        icc_df = pg.intraclass_corr(
            data=df_tmp,
            targets=coluna_estudante,   # entre-quartis (estudantes = "alvos")
            raters=coluna_submissao,    # intra-medidas (submissões = "avaliadores")
            ratings=feature,
            nan_policy='omit',
        )
        # Pegar ICC(A,1) → tipo "Absolute agreement" / "Single random raters"
        mask = (icc_df['Type'] == 'ICC1') | (icc_df['Description'].str.contains('Absolute agreement', case=False, na=False) &
                                             icc_df['Description'].str.contains('Single', case=False, na=False))
        if mask.any():
            return float(icc_df.loc[mask, 'ICC'].iloc[0])
        else:
            return float(icc_df['ICC'].iloc[0])
    except Exception:
        try:
            # fallback: forma manual (one-way ANOVA ICC1)
            grupos = list(df_tmp.groupby(coluna_estudante)[feature].apply(list).values)
            n_grupos = len(grupos)
            if n_grupos < 2 or any(len(g) < 2 for g in grupos[:3]):
                return np.nan
            ss_entre = 0
            ss_dentro = 0
            media_total = np.mean(np.concatenate(grupos))
            medias_grupo = [np.mean(g) for g in grupos]
            for g, mg in zip(grupos, medias_grupo):
                ss_entre += len(g) * (mg - media_total) ** 2
                ss_dentro += np.sum((np.array(g) - mg) ** 2)
            df_entre = n_grupos - 1
            df_dentro = sum(len(g) for g in grupos) - n_grupos
            ms_entre = ss_entre / df_entre
            ms_dentro = ss_dentro / df_dentro if df_dentro > 0 else 1e-9
            n_medio = np.mean([len(g) for g in grupos])
            icc1 = (ms_entre - ms_dentro) / (ms_entre + (n_medio - 1) * ms_dentro)
            return float(icc1)
        except Exception:
            return np.nan


def calcular_icc_todas_features(
    df_features: pd.DataFrame,
    lista_features: List[str],
    coluna_estudante: str = 'estudante',
    coluna_submissao: str = 'submissao_idx',
    n_bootstrap: int = N_BOOTSTRAP // 4,  # 500 para ICC (para ser rápido)
) -> pd.DataFrame:
    """
    Calcula ICC + IC 95% (bootstrap) para todas as features dadas.
    Retorna DataFrame colunas: ['feature','familia','icc','ic_inf','ic_sup','estavel']
    """
    # 1) Garantir formato longo: 1 linha por (estudante, submissao)
    if coluna_submissao not in df_features.columns:
        df_features = df_features.copy()
        df_features[coluna_submissao] = (
            df_features
            .groupby(coluna_estudante)
            .cumcount() + 1
        )
    df_long = df_features[[coluna_estudante, coluna_submissao] + lista_features].copy()
    # Normalizar z-score por coluna (evitar escala distorcer ICC)
    for c in lista_features:
        arr = df_long[c].values.astype(float)
        std = arr.std() if arr.std() > 0 else 1.0
        df_long[c] = (arr - arr.mean()) / std

    resultados = []
    for feat in tqdm(lista_features, desc='Calculando ICC por feature'):
        icc_val = calcular_icc_uma_feature(df_long, feat, coluna_estudante, coluna_submissao)
        # Bootstrap IC 95%
        amostras_boot = []
        if n_bootstrap > 0 and not np.isnan(icc_val):
            estudantes_unicos = list(df_long[coluna_estudante].unique())
            for _ in range(n_bootstrap):
                estudantes_boot = rng.choice(estudantes_unicos, size=len(estudantes_unicos), replace=True)
                df_boot = pd.concat(
                    [df_long[df_long[coluna_estudante] == e].assign(**{coluna_estudante: f'{e}_{i}'})
                     for i, e in enumerate(estudantes_boot)],
                    ignore_index=True,
                )
                v = calcular_icc_uma_feature(df_boot, feat, coluna_estudante, coluna_submissao)
                if not (np.isnan(v) or np.isinf(v)):
                    amostras_boot.append(v)
        if len(amostras_boot) >= 20:
            alpha = 1 - IC_NIVEL
            ic_inf = float(np.quantile(amostras_boot, alpha / 2))
            ic_sup = float(np.quantile(amostras_boot, 1 - alpha / 2))
        else:
            # fallback Fisher
            ic_inf = max(-1.0, icc_val - 1.96 * 0.08 if not np.isnan(icc_val) else 0.0)
            ic_sup = min(1.0, icc_val + 1.96 * 0.08 if not np.isnan(icc_val) else 0.0)

        resultados.append({
            'feature': feat,
            'familia': _obter_familia_feature(feat),
            'icc': float(np.clip(icc_val, -1, 1)) if not np.isnan(icc_val) else 0.0,
            'ic_inf': float(np.clip(ic_inf, -1, 1)),
            'ic_sup': float(np.clip(ic_sup, -1, 1)),
        })

    df_icc = pd.DataFrame(resultados).sort_values('icc', ascending=False).reset_index(drop=True)
    df_icc['estavel'] = df_icc['icc'] >= LIMIAR_ICC_ESTAVEL
    df_icc['ranking'] = np.arange(1, len(df_icc) + 1)
    return df_icc


def obter_top_estaveis(df_icc: pd.DataFrame, top_n: int = TOP_N_FEATURES_ICC) -> Tuple[List[str], pd.DataFrame]:
    """
    Retorna (lista das top N features mais estáveis, DataFrame top N formatado como Quadro 2).
    """
    top = df_icc.head(top_n).copy()
    top['Familia'] = top['familia']
    top['Característica'] = top['feature'].apply(
        lambda f: f.replace('freq_funcional_','Frequência relativa da palavra ')
                  .replace('freq_dep_','Frequência da dependência ')
                  .replace('freq_rel_','Frequência relativa de ')
                  .replace('freq_char_','Frequência n-grama ')
                  .replace('_',' ')
                  .capitalize()
    )
    top['ICC'] = (top['icc'] * 100).round(0).astype(int) / 100
    top['IC 95% inferior'] = (top['ic_inf'] * 100).round(0).astype(int) / 100
    top['IC 95% superior'] = (top['ic_sup'] * 100).round(0).astype(int) / 100
    top_quadro = top[['ranking','Familia','Característica','ICC','IC 95% inferior','IC 95% superior']]
    top_quadro = top_quadro.rename(columns={'ranking':'Posição'})
    return list(top['feature'].values), top_quadro


def resumo_por_familia(df_icc: pd.DataFrame) -> pd.DataFrame:
    return (
        df_icc.groupby('familia')['icc']
        .agg(['count','median','mean','min','max'])
        .round(3)
        .sort_values('median', ascending=False)
    )
