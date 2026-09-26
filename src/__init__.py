"""
Pacote código_artigo — implementação do Artigo Científico:
Perfis estilométricos longitudinais para verificação de autoria no ensino superior.

Módulos:
  - config: constantes e caminhos globais
  - extractor_features: extrai 204 características estilométricas de textos em PT
  - icc_calculator: cálculo do Coeficiente Correlação Intraclasse (estabilidade)
  - perfis: construção de perfis longitudinais vs baseline (trabalho único)
  - modelagem: SVM RBF + validação cruzada aninhada por estudante + métricas
  - avaliacao: gera Figuras 1/2 e Quadros 2/3/4 de resultados
  - corpus_sintetico: gerador de corpus sintético de teste (estudantes + 4 categorias de autoria)
"""

import os
import sys
from pathlib import Path
_PACOTE_DIR = Path(__file__).resolve().parent.parent
_VENDOR = _PACOTE_DIR / '_vendor'
_CACHE = _PACOTE_DIR / '_cache'
_MPL = _CACHE / 'matplotlib'
_NLTK = _CACHE / 'nltk_data'
_MPL.mkdir(parents=True, exist_ok=True)
_NLTK.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(_MPL)
os.environ['NLTK_DATA'] = str(_NLTK)
os.environ['NUMEXPR_MAX_THREADS'] = os.environ.get('NUMEXPR_MAX_THREADS', '8')

if _VENDOR.exists() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
if str(_PACOTE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACOTE_DIR))
if str(_PACOTE_DIR / 'src') not in sys.path:
    sys.path.insert(0, str(_PACOTE_DIR / 'src'))

from .config import (
    SEED, N_BOOTSTRAP, IC_NIVEL,
    LIMIAR_ICC_ESTAVEL, TOP_N_FEATURES_ICC,
    FAMILIAS_FEATURES, PALAVRAS_FUNCIONAIS_PT,
    CORPUS_SINTETICO_DIR, FIGURAS_DIR, TABELAS_DIR,
)
from .extractor_features import extrair_features_texto, extrair_features_corpus, lista_todas_features
from .icc_calculator import (
    calcular_icc_todas_features, obter_top_estaveis, resumo_por_familia,
)
from .perfis import (
    dividir_por_estudante, construir_perfil_longitudinal, construir_perfil_baseline,
    gerar_pares_classificacao,
)
from .modelagem import (
    calcular_metricas, pipeline_svm_nested, comparar_condicoes, desempenho_por_categoria_autoria,
)
from .avaliacao import (
    plotar_figura1_icc_por_familia, plotar_figura2_roc_comparativa,
    guardar_tabela_quadro2, guardar_tabela_quadro3, guardar_tabela_quadro4,
    resumo_texto_resultados,
)
from .corpus_sintetico import gerar_corpus_sintetico

__all__ = [
    'SEED', 'N_BOOTSTRAP', 'IC_NIVEL', 'LIMIAR_ICC_ESTAVEL', 'TOP_N_FEATURES_ICC',
    'FAMILIAS_FEATURES', 'PALAVRAS_FUNCIONAIS_PT',
    'extrair_features_texto', 'extrair_features_corpus', 'lista_todas_features',
    'calcular_icc_todas_features', 'obter_top_estaveis', 'resumo_por_familia',
    'dividir_por_estudante', 'construir_perfil_longitudinal', 'construir_perfil_baseline',
    'gerar_pares_classificacao',
    'calcular_metricas', 'pipeline_svm_nested', 'comparar_condicoes', 'desempenho_por_categoria_autoria',
    'plotar_figura1_icc_por_familia', 'plotar_figura2_roc_comparativa',
    'guardar_tabela_quadro2', 'guardar_tabela_quadro3', 'guardar_tabela_quadro4',
    'resumo_texto_resultados', 'gerar_corpus_sintetico',
    'CORPUS_SINTETICO_DIR', 'FIGURAS_DIR', 'TABELAS_DIR',
]
