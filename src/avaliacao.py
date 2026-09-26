"""
avaliacao.py
============
Gera as figuras e tabelas de resultados do artigo:
  - FIGURA 1: Boxplot da distribuição ICC por família de características (limiar 0,60 tracejado)
  - FIGURA 2: Curvas ROC comparativas (baseline tracejado vs longitudinal contínuo)
  - TABELAS CSV/Excel: Quadro 2 (top 20 ICC), Quadro 3 (comparação métricas), Quadro 4 (desempenho por categoria)
"""

import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Sem GUI
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    FIGURAS_DIR, TABELAS_DIR,
    LIMIAR_ICC_ESTAVEL,
    COR_BASELINE, COR_LONGITUDINAL, CORES_FAMILIAS,
    SEED,
)

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# ====================================================================
# FIGURA 1: Distribuição ICC por família (boxplot)
# ====================================================================
def plotar_figura1_icc_por_familia(df_icc: pd.DataFrame,
                                   nome_arquivo: str = 'Figura1_ICC_por_familia.png',
                                   dpi: int = 300) -> str:
    caminho = FIGURAS_DIR / nome_arquivo
    fig, ax = plt.subplots(figsize=(9, 5.5))
    familias_ordenadas = df_icc.groupby('familia')['icc'].median().sort_values(ascending=False).index.tolist()
    dados_plot = []
    for fam in familias_ordenadas:
        vals = df_icc[df_icc['familia'] == fam]['icc'].tolist()
        for v in vals:
            dados_plot.append({'Família de características': fam, 'ICC (consistência intra-autor)': v})
    df_plot = pd.DataFrame(dados_plot)
    palette = {fam: CORES_FAMILIAS.get(fam, '#888888') for fam in familias_ordenadas}
    sns.boxplot(data=df_plot, x='Família de características', y='ICC (consistência intra-autor)',
                palette=palette, ax=ax, width=0.55, showfliers=False, linewidth=1.5)
    sns.stripplot(data=df_plot, x='Família de características', y='ICC (consistência intra-autor)',
                  color='#222222', alpha=0.28, size=3.5, jitter=0.28, ax=ax)
    # Limiar 0,60
    ax.axhline(LIMIAR_ICC_ESTAVEL, color='#444444', linestyle='--', linewidth=1.5,
               label=f'Limiar de estabilidade (ICC = {LIMIAR_ICC_ESTAVEL:.2f})')
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.10))
    ax.yaxis.grid(True, linestyle=':', color='#BBBBBB', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    ax.set_title('Figura 1. Distribuição do ICC por família de características estilométricas')
    plt.xticks(rotation=12, ha='right')
    plt.tight_layout()
    fig.savefig(caminho, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Figura 1 (ICC) salva: {caminho}")
    return str(caminho)


# ====================================================================
# FIGURA 2: Curvas ROC comparativas baseline vs longitudinal
# ====================================================================
def plotar_figura2_roc_comparativa(
    y_true_baseline: np.ndarray, y_prob_baseline: np.ndarray,
    y_true_longitudinal: np.ndarray, y_prob_longitudinal: np.ndarray,
    nome_arquivo: str = 'Figura2_ROC_comparativa.png',
    dpi: int = 300,
) -> str:
    from sklearn.metrics import roc_curve, auc
    caminho = FIGURAS_DIR / nome_arquivo
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Baseline
    fpr_a, tpr_a, _ = roc_curve(y_true_baseline, y_prob_baseline)
    auc_a = auc(fpr_a, tpr_a)
    ax.plot(fpr_a, tpr_a, color=COR_BASELINE, linestyle='--', linewidth=2.3,
            label=f'Trabalho único (AUC = {auc_a:.2f})')
    # Longitudinal
    fpr_b, tpr_b, _ = roc_curve(y_true_longitudinal, y_prob_longitudinal)
    auc_b = auc(fpr_b, tpr_b)
    ax.plot(fpr_b, tpr_b, color=COR_LONGITUDINAL, linestyle='-', linewidth=2.6,
            label=f'Perfil longitudinal (AUC = {auc_b:.2f})')
    # Linha diagonal (aleatório)
    ax.plot([0, 1], [0, 1], color='#888888', linestyle=':', linewidth=1.3, label='Classificador aleatório (AUC = 0.50)')
    # Área de melhoria FPR < 0.20
    ax.fill_betweenx([0, 1], 0, 0.20, color='#CCCCCC', alpha=0.25,
                     label='Região de interesse prático (FPR < 20%)')
    # Zoom ou destaque
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Taxa de falsos positivos (FPR)')
    ax.set_ylabel('Taxa de verdadeiros positivos (TPR / Revocação)')
    ax.set_title('Figura 2. Curvas ROC comparativas entre abordagens de verificação de autoria')
    ax.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.92)
    ax.grid(True, linestyle=':', color='#BBBBBB', alpha=0.5)
    # Inset zoom em FPR < 0.2
    ax_inset = ax.inset_axes([0.52, 0.12, 0.45, 0.45])
    ax_inset.plot(fpr_a, tpr_a, color=COR_BASELINE, linestyle='--', linewidth=2)
    ax_inset.plot(fpr_b, tpr_b, color=COR_LONGITUDINAL, linestyle='-', linewidth=2.3)
    ax_inset.set_xlim(0.0, 0.20)
    ax_inset.set_ylim(0.3, 1.02)
    ax_inset.set_xticks([0, 0.05, 0.1, 0.15, 0.20])
    ax_inset.set_yticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax_inset.grid(True, linestyle=':', color='#BBBBBB', alpha=0.5)
    ax_inset.set_xlabel('FPR')
    ax_inset.set_ylabel('TPR')
    ax.indicate_inset_zoom(ax_inset, edgecolor='#333333', linewidth=1.2)
    plt.tight_layout()
    fig.savefig(caminho, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Figura 2 (ROC) salva: {caminho}")
    return str(caminho)


# ====================================================================
# Guardar tabelas (CSV + Excel)
# ====================================================================
def guardar_tabela_quadro2(df_quadro2: pd.DataFrame, nome: str = 'Quadro2_Top20_ICC_Estaveis') -> str:
    caminho_csv = TABELAS_DIR / f'{nome}.csv'
    caminho_xlsx = TABELAS_DIR / f'{nome}.xlsx'
    df_quadro2.to_csv(caminho_csv, index=False, sep=';', encoding='utf-8-sig')
    try:
        df_quadro2.to_excel(caminho_xlsx, index=False)
        print(f"[OK] Quadro 2 (Top 20 ICC): CSV={caminho_csv} | XLSX={caminho_xlsx}")
    except Exception as e:
        print(f"[WARN] XLSX falhou: {e}. CSV OK.")
    return str(caminho_csv)


def guardar_tabela_quadro3(df_quadro3: pd.DataFrame, nome: str = 'Quadro3_Comparacao_Metricas') -> str:
    caminho_csv = TABELAS_DIR / f'{nome}.csv'
    caminho_xlsx = TABELAS_DIR / f'{nome}.xlsx'
    df_quadro3.to_csv(caminho_csv, sep=';', encoding='utf-8-sig')
    try:
        df_quadro3.to_excel(caminho_xlsx)
        print(f"[OK] Quadro 3 (comparação métricas): CSV={caminho_csv} | XLSX={caminho_xlsx}")
    except Exception as e:
        print(f"[WARN] XLSX falhou: {e}. CSV OK.")
    return str(caminho_csv)


def guardar_tabela_quadro4(df_quadro4: pd.DataFrame, nome: str = 'Quadro4_Desempenho_por_Categoria') -> str:
    caminho_csv = TABELAS_DIR / f'{nome}.csv'
    caminho_xlsx = TABELAS_DIR / f'{nome}.xlsx'
    df_quadro4.to_csv(caminho_csv, sep=';', encoding='utf-8-sig')
    try:
        df_quadro4.to_excel(caminho_xlsx)
        print(f"[OK] Quadro 4 (desempenho por categoria): CSV={caminho_csv} | XLSX={caminho_xlsx}")
    except Exception as e:
        print(f"[WARN] XLSX falhou: {e}. CSV OK.")
    return str(caminho_csv)


def resumo_texto_resultados(
    metricas_baseline: Dict[str, float], metricas_long: Dict[str, float],
    df_quadro3: pd.DataFrame,
) -> str:
    """Gera um sumário textual alinhado com o §4.2 do artigo."""
    auc_a = metricas_baseline.get('auc_roc', np.nan)
    auc_b = metricas_long.get('auc_roc', np.nan)
    fpr_a = metricas_baseline.get('fpr', np.nan) * 100
    fpr_b = metricas_long.get('fpr', np.nan) * 100
    prec_a = metricas_baseline.get('precision', np.nan) * 100
    prec_b = metricas_long.get('precision', np.nan) * 100
    rec_a = metricas_baseline.get('recall', np.nan) * 100
    rec_b = metricas_long.get('recall', np.nan) * 100
    delta_auc = (auc_b - auc_a) * 100
    delta_fpr = fpr_b - fpr_a
    reducao_rel = (fpr_a - fpr_b) / max(1.0, fpr_a) * 100

    return (
        f"=== Sumário Resultados §4.2 (Comparação Baseline vs Longitudinal) ===\n"
        f"  AUC-ROC: Baseline {auc_a:.2%} vs Longitudinal {auc_b:.2%}  (Δ +{delta_auc:.1f} p.p.)\n"
        f"  FPR:     Baseline {fpr_a:.1f}% vs Longitudinal {fpr_b:.1f}%  (Δ {delta_fpr:+.1f} p.p.) → "
        f"Redução relativa ≈ {reducao_rel:.0f}%\n"
        f"  Precisão: Baseline {prec_a:.1f}% vs Longitudinal {prec_b:.1f}%  (Δ +{prec_b - prec_a:+.1f} p.p.)\n"
        f"  Revocação: Baseline {rec_a:.1f}% vs Longitudinal {rec_b:.1f}%  (Δ {rec_b - rec_a:+.1f} p.p.)\n"
        f"\nValores alvo esperados (artigo V2):\n"
        f"  AUC Δ ≈ +10 p.p. | FPR Δ ≈ -22 p.p. (redução ~71%) | Precisão Δ ≈ +22 p.p. | Revocação Δ ≈ 0 p.p."
    )
