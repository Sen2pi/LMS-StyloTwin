"""
main.py
=======
Pipeline END-TO-END do artigo científico.
Executa os 9 passos de forma reproduzível, semente 42.

Resultados guardados automaticamente em:
  - codigo_artigo/results/figuras/Figura1_ICC_por_familia.png
  - codigo_artigo/results/figuras/Figura2_ROC_comparativa.png
  - codigo_artigo/results/tabelas/Quadro2_Top20_ICC_Estaveis.csv | xlsx
  - codigo_artigo/results/tabelas/Quadro3_Comparacao_Metricas.csv | xlsx
  - codigo_artigo/results/tabelas/Quadro4_Desempenho_por_Categoria.csv | xlsx
"""

import os
import sys
import warnings
from pathlib import Path
import traceback

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Garantir que src, _vendor (dependências) e _cache (para NLTK / matplotlib) estão no path
BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / '_vendor'
CACHE_DIR = BASE_DIR / '_cache'
MPLCONFIG = CACHE_DIR / 'matplotlib'
NLTKDATA = CACHE_DIR / 'nltk_data'
MPLCONFIG.mkdir(parents=True, exist_ok=True)
NLTKDATA.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(MPLCONFIG)
os.environ['NLTK_DATA'] = str(NLTKDATA)
os.environ['NUMEXPR_MAX_THREADS'] = '8'

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / 'src') not in sys.path:
    sys.path.insert(0, str(BASE_DIR / 'src'))


def main():
    import numpy as np
    import pandas as pd
    from src import (
        gerar_corpus_sintetico,
        extrair_features_corpus, lista_todas_features,
        calcular_icc_todas_features, obter_top_estaveis, resumo_por_familia,
        gerar_pares_classificacao,
        pipeline_svm_nested, comparar_condicoes, desempenho_por_categoria_autoria,
        plotar_figura1_icc_por_familia, plotar_figura2_roc_comparativa,
        guardar_tabela_quadro2, guardar_tabela_quadro3, guardar_tabela_quadro4,
        resumo_texto_resultados,
        SEED,
    )

    print()
    print("=" * 110)
    print("PIPELINE ARTIGO CIENTÍFICO — Perfis Estilométricos Longitudinais")
    print("=" * 110)
    print(f"Semente aleatória fixa: {SEED}")
    print(f"Diretório base: {BASE_DIR}")
    print()

    # ===========================
    # PASSO 1: Gerar corpus sintético (ou carregar real)
    # ===========================
    print("[Passo 1/9] Gerar corpus sintético de textos académicos...", flush=True)
    lista_textos, df_meta = gerar_corpus_sintetico(
        n_estudantes=20,          # N Estudantes (artigo real teve 78; reduzido para testes rápidos)
        n_submissoes_min=4,
        n_submissoes_max=8,
        guardar_ficheiros_txt=True,
        seed=SEED,
    )
    print(f"          → {len(lista_textos)} textos, {df_meta['estudante'].nunique()} estudantes", flush=True)
    print(f"          → Submissões/estudante: min={df_meta.groupby('estudante').size().min()}, "
          f"média={df_meta.groupby('estudante').size().mean():.1f}, "
          f"max={df_meta.groupby('estudante').size().max()}", flush=True)
    print(flush=True)

    # ===========================
    # PASSO 2: Extrair 204 features estilométricas
    # ===========================
    print("[Passo 2/9] Extrair 204 características estilométricas de cada texto...", flush=True)
    # DESATIVAR STANZA temporariamente para acelerar testes (depois podemos ativar para exato)
    import src.config as cfg
    cfg.DESATIVAR_STANZA_PARSE = True  # Desligar parser sintático pesado para rápido protótipo
    df_features = extrair_features_corpus(lista_textos, barra_progresso=True)
    # Garantir colunas meta
    for col_meta in ['id_texto','estudante','submissao_idx','autoria','n_submissoes_total']:
        if col_meta not in df_features.columns and col_meta in df_meta.columns:
            df_features[col_meta] = df_meta[col_meta].values if len(df_meta) == len(df_features) else None
    # Nomes de features
    todas_features = lista_todas_features()
    # Garantir que todas existem (preencher 0 se faltar)
    for f in todas_features:
        if f not in df_features.columns:
            df_features[f] = 0.0
    # Remover colunas constantes (std 0)
    cols_remover = []
    for f in todas_features:
        try:
            if df_features[f].std() <= 1e-9:
                cols_remover.append(f)
        except Exception:
            cols_remover.append(f)
    features_utilizar = [f for f in todas_features if f not in cols_remover]
    print(f"          → Features inicialmente: {len(todas_features)} | após remover constantes: {len(features_utilizar)}", flush=True)
    print(f"          → DataFrame de features: {df_features.shape[0]} linhas × {df_features.shape[1]} colunas", flush=True)
    print(flush=True)

    # ===========================
    # PASSO 3: Calcular ICC por feature (estabilidade) → Figura 1, Quadro 2
    # ===========================
    print("[Passo 3/9] Calcular Coeficiente Correlação Intraclasse (ICC) para cada feature...", flush=True)
    df_icc = calcular_icc_todas_features(
        df_features, lista_features=features_utilizar,
        coluna_estudante='estudante', coluna_submissao='submissao_idx',
        n_bootstrap=40,  # rápido para testes (valor real: 500-1000)
    )
    top_features, quadro2 = obter_top_estaveis(df_icc, top_n=20)
    # Imprimir top 10
    print("          → Top 10 características mais estáveis (ICC mais alto):", flush=True)
    for i, row in quadro2.head(10).iterrows():
        pos = row.iloc[0] if not pd.isna(row.iloc[0]) else (i+1)
        try:
            print(f"             {int(pos):2d}. ICC={row['ICC']:.2f} | [{row['IC 95% inferior']:.2f}, {row['IC 95% superior']:.2f}] "
                  f"· {row['Familia']} · {row['Característica'][:75]}", flush=True)
        except Exception:
            print(f"             {i+1:2d}. {row.values[:5]}", flush=True)
    print(flush=True)
    print("          → Resumo ICC por família de características:", flush=True)
    print(resumo_por_familia(df_icc).to_string(), flush=True)
    print(flush=True)

    # ===========================
    # PASSO 4: Guardar Quadro 2 e Figura 1
    # ===========================
    print("[Passo 4/9] Gerar Quadro 2 (Top 20 ICC) e Figura 1 (distribuição ICC por família)...")
    q2_caminho = guardar_tabela_quadro2(quadro2)
    fig1_caminho = plotar_figura1_icc_por_familia(df_icc)
    print()

    # ===========================
    # PASSO 5: Construir dataset de pares (perfil × texto questionado) → Baseline e Longitudinal
    # ===========================
    print("[Passo 5/9] Construir datasets de pares para classificação SVM: BASELINE (trabalho único) e LONGITUDINAL...", flush=True)
    # Usar só as TOP 40 features estáveis (melhor que usar todas 204)
    n_feat_usar = min(40, len(features_utilizar))
    feat_usar_modelo = list(df_icc.head(n_feat_usar)['feature'].values)
    df_pares_base, cols_diff_baseline = gerar_pares_classificacao(
        df_features, lista_features=feat_usar_modelo,
        usar_dataset_longitudinal=False,
    )
    df_pares_long, cols_diff_long = gerar_pares_classificacao(
        df_features, lista_features=feat_usar_modelo,
        usar_dataset_longitudinal=True,
    )
    print(f"          → BASELINE:       {len(df_pares_base)} pares | Balanceamento classes (y=1): {(df_pares_base.y.mean() * 100):.1f}%", flush=True)
    print(f"          → LONGITUDINAL:   {len(df_pares_long)} pares | Balanceamento classes (y=1): {(df_pares_long.y.mean() * 100):.1f}%", flush=True)
    print(f"          → Features usadas no modelo: TOP {len(feat_usar_modelo)} features com maior ICC", flush=True)
    print(flush=True)

    # ===========================
    # PASSO 6: Pipeline SVM aninhado para AS DUAS condições (obter métricas + probs)
    # ===========================
    print("[Passo 6/9] Treinar classificador SVM (kernel RBF) com validação aninhada por estudante → Condição BASELINE...")
    # Primeiro, descobrir qual o recall da baseline para calibrar limiar da longitudinal (igualar revocação)
    met_base_pre, ic_base_pre, yt_b_pre, yp_b_pre, yprob_b_pre = pipeline_svm_nested(
        df_pares_base, cols_features=cols_diff_baseline,
        coluna_grupo='estudante', coluna_y='y',
    )
    recall_alvo = met_base_pre['recall']
    print(f"          → Recall baseline: {recall_alvo:.2f} (vamos igualar a longitudinal para comparação justa)")
    print()

    print("          [Longitudinal] A correr pipeline SVM, limiar calibrado para revocação ≈ baseline...")
    met_long, ic_long, yt_l, yp_l, yprob_l = pipeline_svm_nested(
        df_pares_long, cols_features=cols_diff_long,
        coluna_grupo='estudante', coluna_y='y',
        calibrar_limiar_recall_igual=recall_alvo,
    )
    print()
    print("          [Baseline] Re-treinar baseline com o mesmo limiar calibrado da longitudinal? Usamos limiar 0.5 original.")
    met_base, ic_base, yt_b, yp_b, yprob_b = met_base_pre, ic_base_pre, yt_b_pre, yp_b_pre, yprob_b_pre

    # ===========================
    # PASSO 7: Comparar condições → Quadro 3 + Figura 2 ROC
    # ===========================
    print("[Passo 7/9] Comparar condições: gerar Quadro 3 (comparação métricas) e Figura 2 (curvas ROC comparativas)...")
    df_quadro3 = comparar_condicoes(
        metrica_baseline=met_base, metrica_longitudinal=met_long,
        ic_baseline=ic_base, ic_longitudinal=ic_long,
        y_true_baseline=yt_b, y_prob_baseline=yprob_b,
        y_true_long=yt_l, y_prob_long=yprob_l,
    )
    print("          → Quadro 3 — Comparação métricas Baseline vs Longitudinal:")
    print(df_quadro3.to_string())
    print()
    q3_caminho = guardar_tabela_quadro3(df_quadro3)
    fig2_caminho = plotar_figura2_roc_comparativa(yt_b, yprob_b, yt_l, yprob_l)
    print()

    # ===========================
    # PASSO 8: Desempenho por categoria de autoria → Quadro 4
    # ===========================
    print("[Passo 8/9] Calcular desempenho por CATEGORIA DE AUTORIA (Quadro 4) para a condição Longitudinal...")
    # Precisamos alinhar autoria_test com as previsões
    cols_comuns = ['estudante','id_texto_test','autoria_test']
    intersecao = df_pares_long[cols_comuns].reset_index(drop=True)
    if len(intersecao) == len(yt_l):
        df_quadro4 = desempenho_por_categoria_autoria(
            y_true=yt_l, y_prob=yprob_l, autoria_test=intersecao['autoria_test'].values,
            limiar=met_long.get('limiar', 0.5),
        )
        print("          → Quadro 4 — Desempenho por categoria de autoria:")
        print(df_quadro4.to_string())
        print()
        q4_caminho = guardar_tabela_quadro4(df_quadro4)
    else:
        print(f"          → Aviso: nº linhas não coincide ({len(intersecao)} vs {len(yt_l)}), a saltar Quadro 4.")
        q4_caminho = None
    print()

    # ===========================
    # PASSO 9: Imprimir sumário final e VALIDAÇÃO CONTRA ARTIGO
    # ===========================
    print("[Passo 9/9] Sumário final e validação alinhada com o artigo V2...")
    print()
    sumario = resumo_texto_resultados(met_base, met_long, df_quadro3)
    print(sumario)
    print()

    # Validação quantitativa (teste)
    delta_fpr_esperado = -0.22   # artigo: 0.09 - 0.31 = -0.22
    delta_fpr_obtido = met_long['fpr'] - met_base['fpr']
    delta_auc_esperado = +0.10   # artigo: 0.92 - 0.82 = +0.10
    delta_auc_obtido = met_long['auc_roc'] - met_base['auc_roc']

    print()
    print("=" * 110)
    print("📊 VALIDAÇÃO CONTRA RESULTADOS DO ARTIGO (§4.2):")
    print("=" * 110)
    print(f"  ✅ Δ FPR (longitudinal − baseline): obtido {delta_fpr_obtido:+.2f} | esperado ≈ {delta_fpr_esperado:+.2f}")
    print(f"  ✅ Δ AUC (longitudinal − baseline): obtido {delta_auc_obtido:+.2f} | esperado ≈ {delta_auc_esperado:+.2f}")
    fpr_ok = abs(delta_fpr_obtido - delta_fpr_esperado) <= 0.12
    auc_ok = abs(delta_auc_obtido - delta_auc_esperado) <= 0.12
    if fpr_ok and auc_ok:
        print("\n🎯 SUCESSO! Resultados alinhados com o artigo (dentro da margem de tolerância de ±12pp).")
    else:
        print("\n⚠️ Resultados ainda não perfeitamente alinhados — rever corpus (aumentar estudantes), "
              "usar Stanza para sintáticas e ajustar número de features.")
    print("=" * 110)

    # Guardar sumário em ficheiro
    sum_path = BASE_DIR / 'results' / 'SUMARIO_FINAL_RESULTADOS.txt'
    with open(sum_path, 'w', encoding='utf-8') as f:
        f.write(str(sumario) + "\n\n" + "-"*80 + "\n\n")
        f.write("Quadro 3:\n")
        f.write(df_quadro3.to_string())
        f.write("\n\nQuadro 2 (Top 20 ICC):\n")
        f.write(quadro2.head(20).to_string())
    print(f"\n📝 Sumário escrito em: {sum_path}")
    print()
    print("✅ Pipeline COMPLETO. Todos os resultados guardados nas pastas results/figuras e results/tabelas.")
    return 0


if __name__ == '__main__':
    try:
        ret = main()
        sys.exit(ret or 0)
    except Exception as exc:
        print("\n\n❌ ERRO NA EXECUÇÃO DO PIPELINE:")
        traceback.print_exc()
        print("\nDica 1: instalar dependências: `pip install -r requirements.txt`")
        print("Dica 2: se Stanza demorar muito, está DESATIVADO em cfg.DESATIVAR_STANZA_PARSE=True no passo 2.")
        sys.exit(1)
