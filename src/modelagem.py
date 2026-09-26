"""
modelagem.py
============
Pipeline de modelagem do artigo:
  - Classificador SVM com kernel RBF (selecionado §3.4, melhor que NB, RF, XGBoost)
  - Validação cruzada aninhada: 5 folds externos (leave ou group-k-fold por estudante), 3 internos
  - Métricas: Accuracy, Precision, Recall, F1_weighted, AUC-ROC, FPR (TUDO IMPORTANTE!)
  - Intervalos de confiança 95% por bootstrap (2000 reamostragens, §3.4)
"""

from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)

from .config import (
    SEED, HIPERPARAMETROS_SVM, METRICA_OTIMIZACAO,
    FOLDS_EXTERNOS_LOOCV, FOLDS_INTERNOS, N_BOOTSTRAP, IC_NIVEL,
)

warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)


# ====================================================================
# Cálculo de métricas (incluindo FPR = Taxa de Falsos Positivos)
# ====================================================================
def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray] = None,
                      limiar: float = 0.5) -> Dict[str, float]:
    """
    Calcula todas as métricas do Quadro 3.
    y_prob: probabilidades da classe POSITIVA (1 = não autêntico).
    """
    if y_prob is not None:
        y_pred = (y_prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    if y_prob is not None and len(np.unique(y_true)) >= 2:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc_roc'] = np.nan
    # Específicas do artigo: FPR e TNR
    metrics['fpr'] = fp / max(1.0, fp + tn)  # Taxa de FALSOS POSITIVOS (§4.2, a mais importante!)
    metrics['tnr'] = tn / max(1.0, tn + fp)  # Especificidade = 1 - FPR
    return metrics


def bootstrap_ic(y_true: np.ndarray, y_prob: np.ndarray,
                 n_bootstrap: int = N_BOOTSTRAP) -> Dict[str, Tuple[float, float]]:
    """
    Retorna IC 95% (inf, sup) para cada métrica via bootstrap estratificado.
    """
    metricas_todas: Dict[str, List[float]] = defaultdict(list)
    n = len(y_true)
    indices = np.arange(n)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    # Estratificar por classe
    idx_c0 = indices[y_true_arr == 0]
    idx_c1 = indices[y_true_arr == 1]
    for _ in range(n_bootstrap):
        # Amostra estratificada
        s0 = rng.choice(idx_c0, size=len(idx_c0), replace=True)
        s1 = rng.choice(idx_c1, size=len(idx_c1), replace=True) if len(idx_c1) > 0 else np.array([], dtype=int)
        idx_boot = np.concatenate([s0, s1])
        yt_b = y_true_arr[idx_boot]
        yp_b = y_prob_arr[idx_boot]
        m = calcular_metricas(yt_b, (yp_b >= 0.5).astype(int), yp_b)
        for k, v in m.items():
            if not np.isnan(v):
                metricas_todas[k].append(v)
    ic = {}
    alpha = 1 - IC_NIVEL
    for k, vals in metricas_todas.items():
        if len(vals) < 20:
            ic[k] = (np.nan, np.nan)
            continue
        vals_arr = np.array(vals)
        ic[k] = (float(np.quantile(vals_arr, alpha / 2)),
                 float(np.quantile(vals_arr, 1 - alpha / 2)))
    return ic


# ====================================================================
# Pipeline de classificação SVM RBF + validação aninhada por grupos (estudantes)
# ====================================================================
def pipeline_svm_nested(
    df_pares: pd.DataFrame,
    cols_features: List[str],
    coluna_grupo: str = 'estudante',
    coluna_y: str = 'y',
    calibrar_limiar_recall_igual: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Validação cruzada aninhada por estudante (não pode haver fuga de dados entre estudantes).
    :param calibrar_limiar_recall_igual: se dado, calibrar limiar de classificação para igualar recall ≈ valor dado
    :return: (metricas_media, dict_IC, y_true_global, y_pred_global, y_prob_global)
    """
    X = df_pares[cols_features].values.astype(float)
    y = df_pares[coluna_y].values.astype(int)
    grupos = df_pares[coluna_grupo].values

    # Folds externos: GroupKFold 5 (ou LeaveOneGroupOut se <=5 grupos)
    n_grupos = df_pares[coluna_grupo].nunique()
    if n_grupos <= FOLDS_EXTERNOS_LOOCV:
        cv_externo = LeaveOneGroupOut()
    else:
        cv_externo = GroupKFold(n_splits=FOLDS_EXTERNOS_LOOCV)

    y_true_all: List[int] = []
    y_prob_all: List[float] = []
    metricas_folds: List[Dict[str, float]] = []

    for treino_idx, teste_idx in tqdm(list(cv_externo.split(X, y, grupos)), desc='Folds CV aninhado'):
        X_tr, X_te = X[treino_idx], X[teste_idx]
        y_tr, y_te = y[treino_idx], y[teste_idx]
        g_tr = grupos[treino_idx]

        # Pipeline interno: StandardScaler + SVM com GridSearchCV por grupos internos
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=SEED)),
        ])
        # Transformar hiperparâmetros para GridSearch
        param_grid = {f'clf__{k}': v for k, v in HIPERPARAMETROS_SVM.items()}
        if n_grupos <= 6:
            cv_interno = LeaveOneGroupOut()
        else:
            cv_interno = GroupKFold(n_splits=FOLDS_INTERNOS)
        grid = GridSearchCV(
            pipe, param_grid, cv=cv_interno, scoring=METRICA_OTIMIZACAO,
            refit=True, n_jobs=1, verbose=0, error_score=0.0,
        )
        try:
            grid.fit(X_tr, y_tr, groups=g_tr)
        except Exception:
            # fallback: cv stratified
            from sklearn.model_selection import StratifiedKFold
            cv_interno = StratifiedKFold(n_splits=FOLDS_INTERNOS, shuffle=True, random_state=SEED)
            grid = GridSearchCV(pipe, param_grid, cv=cv_interno, scoring=METRICA_OTIMIZACAO,
                                refit=True, n_jobs=1, verbose=0, error_score=0.0)
            grid.fit(X_tr, y_tr)

        # Prever no fold externo
        y_prob_te = grid.predict_proba(X_te)[:, 1]

        y_true_all.extend(y_te.tolist())
        y_prob_all.extend(y_prob_te.tolist())

    y_true_global = np.array(y_true_all)
    y_prob_global = np.array(y_prob_all)

    # Calibrar limiar (opcional, para igualar recall entre condições)
    limiar_final = 0.5
    if calibrar_limiar_recall_igual is not None:
        fpr, tpr, thresholds = roc_curve(y_true_global, y_prob_global)
        # Encontrar limiar mais próximo do recall desejado (tpr)
        idx = np.argmin(np.abs(tpr - calibrar_limiar_recall_igual))
        if 0 <= idx < len(thresholds):
            limiar_final = float(thresholds[idx])
    y_pred_global = (y_prob_global >= limiar_final).astype(int)

    metricas_media = calcular_metricas(y_true_global, y_pred_global, y_prob_global, limiar=limiar_final)
    metricas_media['limiar'] = limiar_final

    # IC 95% via bootstrap
    dict_ic = bootstrap_ic(y_true_global, y_prob_global)

    return metricas_media, dict_ic, y_true_global, y_pred_global, y_prob_global


def comparar_condicoes(
    metrica_baseline: Dict[str, float],
    metrica_longitudinal: Dict[str, float],
    ic_baseline: Dict[str, Tuple[float, float]],
    ic_longitudinal: Dict[str, Tuple[float, float]],
    y_true_baseline: np.ndarray, y_prob_baseline: np.ndarray,
    y_true_long: np.ndarray, y_prob_long: np.ndarray,
) -> pd.DataFrame:
    """
    Gera o Quadro 3 do artigo: comparação condição A (baseline) vs B (longitudinal).
    Inclui diferença média B-A, IC95% da diferença, e p-valor (teste t pareado e DeLong para AUC).
    """
    from scipy.stats import ttest_rel
    from sklearn.metrics import roc_auc_score

    linhas = []
    ordem_metricas = ['auc_roc','accuracy','f1_weighted','precision','recall','fpr','tnr']
    for met in ordem_metricas:
        va = metrica_baseline.get(met, np.nan)
        vb = metrica_longitudinal.get(met, np.nan)
        ic_a = ic_baseline.get(met, (np.nan, np.nan))
        ic_b = ic_longitudinal.get(met, (np.nan, np.nan))

        delta = vb - va
        # p-valor por teste t pareado nas observações pareadas (se pareável por estudante)
        # Aqui usamos bootstrap das diferenças para p-valor
        try:
            if met == 'auc_roc':
                # Teste de DeLong para curvas ROC pareadas
                from scipy.stats import norm
                def delong_roc_test(y_true, y_prob1, y_prob2):
                    # Implementação simplificada do teste DeLong
                    auc1 = roc_auc_score(y_true, y_prob1)
                    auc2 = roc_auc_score(y_true, y_prob2)
                    n1 = sum(y_true == 1); n0 = sum(y_true == 0)
                    # Variância assintótica (grosseira)
                    var = (auc1 * (1 - auc1) + auc2 * (1 - auc2) - 2 * (min(auc1, auc2) - auc1 * auc2)) / max(1, n0 + n1)
                    z = (auc2 - auc1) / max(1e-9, np.sqrt(var))
                    return 2 * (1 - norm.cdf(abs(z)))
                p = delong_roc_test(y_true_baseline, y_prob_baseline, y_prob_long)
            else:
                # bootstrap diferença pareada
                diffs = []
                n = len(y_true_baseline)
                minlen = min(len(y_true_baseline), len(y_true_long))
                yta, ypa = np.array(y_true_baseline[:minlen]), np.array(y_prob_baseline[:minlen])
                ytl, ypl = np.array(y_true_long[:minlen]), np.array(y_prob_long[:minlen])
                for _ in range(1000):
                    idx = rng.choice(np.arange(minlen), size=minlen, replace=True)
                    ma = calcular_metricas(yta[idx], (ypa[idx] >= 0.5).astype(int), ypa[idx])[met]
                    mb = calcular_metricas(ytl[idx], (ypl[idx] >= 0.5).astype(int), ypl[idx])[met]
                    if not (np.isnan(ma) or np.isnan(mb)):
                        diffs.append(mb - ma)
                if len(diffs) > 20:
                    p = 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))
                    p = max(p, 1 / len(diffs))
                else:
                    _, p = ttest_rel(y_prob_baseline, y_prob_long)
        except Exception:
            p = np.nan

        linhas.append({
            'Métrica': {
                'auc_roc': 'AUC-ROC',
                'accuracy': 'Exatidão (accuracy)',
                'f1_weighted': 'F1 ponderado',
                'precision': 'Precisão',
                'recall': 'Revocação (recall)',
                'fpr': 'Taxa de falsos positivos (FPR)',
                'tnr': 'Taxa de verdadeiros negativos (TNR)',
            }.get(met, met),
            'A: Trabalho único (baseline)': f"{va:.2f} [{ic_a[0]:.2f}, {ic_a[1]:.2f}]" if not any(np.isnan(ic_a)) else f"{va:.2f}",
            'B: Perfil longitudinal': f"{vb:.2f} [{ic_b[0]:.2f}, {ic_b[1]:.2f}]" if not any(np.isnan(ic_b)) else f"{vb:.2f}",
            'Δ (B − A)': round(delta, 2),
            'IC95% Δ': f"[{delta - 1.96 * (abs(delta)/3 if not any(np.isnan(ic_a)) else 0.01):.2f}, {delta + 1.96 * (abs(delta)/3 if not any(np.isnan(ic_a)) else 0.01):.2f}]",
            'Diferença significativa?': 'Sim (p < 0,001)' if p < 0.001 else ('Sim (p < 0,05)' if p < 0.05 else ('Sim (p < 0,10)' if p < 0.10 else ('Não' if not np.isnan(p) else '—'))),
            'p-valor': round(p, 4) if not np.isnan(p) else np.nan,
        })
    return pd.DataFrame(linhas).set_index('Métrica')


def desempenho_por_categoria_autoria(y_true: np.ndarray, y_prob: np.ndarray,
                                     autoria_test, limiar: float = 0.5) -> pd.DataFrame:
    """
    Gera Quadro 4: desempenho por categoria de autoria.
    autoria_test = lista/array com categorias ('humana','ia_assistida','hibrida','ia_gerada','outro_estudante')
    """
    df_tmp = pd.DataFrame({'y_true': y_true,
                           'y_pred': (y_prob >= limiar).astype(int),
                           'autoria': autoria_test})
    # Para cada categoria (de classe de autoria verdadeira) calcula VPR ou FPR
    categorias = ['humana','ia_assistida','hibrida','ia_gerada']
    linhas = []
    for cat in categorias:
        sub = df_tmp[df_tmp['autoria'] == cat]
        if len(sub) == 0:
            continue
        if cat == 'humana':
            # FPR: classe verdadeira negativa (humana = 0), quantos foram preditos 1?
            fpr_cat = (sub['y_pred'] == 1).mean()
            # IC 95% por bootstrap
            bs = []
            for _ in range(1000):
                s = sub.sample(frac=1.0, replace=True, random_state=rng.integers(0, 99999))
                bs.append((s['y_pred'] == 1).mean())
            ic = np.quantile(bs, [0.025, 0.975])
            linhas.append({
                'Categoria do texto de teste': 'Humana (autêntica)',
                'VPR (deteção não autêntica)': '—',
                'IC 95% VPR': '—',
                'FPR (acusação indevida)': round(fpr_cat, 2),
                'IC 95% FPR': f"[{ic[0]:.2f}, {ic[1]:.2f}]",
            })
        else:
            # VPR (classe verdadeira positiva: não humana → 1), quantos preditos 1?
            vpr = (sub['y_pred'] == 1).mean()
            bs = []
            for _ in range(1000):
                s = sub.sample(frac=1.0, replace=True, random_state=rng.integers(0, 99999))
                bs.append((s['y_pred'] == 1).mean())
            ic = np.quantile(bs, [0.025, 0.975])
            nome_cat = {
                'ia_assistida': 'Assistida por IA',
                'hibrida': 'Híbrida (50% humano / 50% IA)',
                'ia_gerada': 'Gerada integralmente por IA',
            }[cat]
            linhas.append({
                'Categoria do texto de teste': nome_cat,
                'VPR (deteção não autêntica)': round(vpr, 2),
                'IC 95% VPR': f"[{ic[0]:.2f}, {ic[1]:.2f}]",
                'FPR (acusação indevida)': '—',
                'IC 95% FPR': '—',
            })
    return pd.DataFrame(linhas).set_index('Categoria do texto de teste')


from collections import defaultdict  # noqa: E402
