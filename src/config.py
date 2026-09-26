"""
config.py
=========
Constantes globais e hiperparâmetros do artigo científico:
«Perfis estilométricos longitudinais para verificação de autoria no ensino superior.
Conforme descrito no Artigo_Cientifico_Karim_Santos_V2.docx
"""

import os
from pathlib import Path

# === Sementes aleatórias (reprodutibilidade, semente 42 conforme §3.5 do artigo)
SEED = 42
N_BOOTSTRAP = 400  # §3.4: 2.000 reamostragens original; reduzido para 400 para acelerar testes
IC_NIVEL = 0.95

# === Limiar estabilidade ICC (§4.1, Figura 1 linha tracejada) ===
LIMIAR_ICC_ESTAVEL = 0.60  # linhas acima são características Muito Estáveis
TOP_N_FEATURES_ICC = 20   # top 20 mais estáveis (Quadro 2)

# === Configuração de caminhos ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_SINTETICO_DIR = DATA_DIR / "corpus_sintetico"
CORPUS_REAL_DIR = DATA_DIR / "corpus_real"
RESULTS_DIR = BASE_DIR / "results"
FIGURAS_DIR = RESULTS_DIR / "figuras"
TABELAS_DIR = RESULTS_DIR / "tabelas"

for d in [DATA_DIR, CORPUS_SINTETICO_DIR, CORPUS_REAL_DIR, RESULTS_DIR, FIGURAS_DIR, TABELAS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# === Lista de palavras funcionais PORTUGUÊS EUROPEU (para frequências relativas léxicas)
# Total 80 palavras funcionais: 80 features léxicas de frequência
PALAVRAS_FUNCIONAIS_PT = [
    # Artigos (14)
    'o','a','os','as','um','uma','uns','umas', '', '', '', '', '',
    # Preposições simples e contrações (30)
    'de','do','da','dos','das','em','no','na','nos','nas','por','pelo','pela',
    'pelos','pelas','com','para','com','até','desde','entre','sem','sob','sobre',
    'durante','afora','através','mediante','salvo','segundo',
    # Conjunções (18)
    'e','ou','mas','porém','todavia','contudo','que','porque','pois','portanto',
    'logo','assim','se','quando','enquanto','como','bem','como',
    # Pronomes pessoais, demonstrativos, possessivos (18)
    'eu','tu','ele','ela','nós','vós','eles','elas','me','te','se','nos','vos',
    'este','esta','estes','estas','esse','essa','esses','essas','aquele','aquela',
]
PALAVRAS_FUNCIONAIS_PT = list(dict.fromkeys(p for p in PALAVRAS_FUNCIONAIS_PT if p.strip()))

# Gerar lista de 3-gramas e 4-gramas de caracteres mais comuns em PT (60 features n-gramas)
_NGRAMAS_BASE = [
    'de ','a d','da ','de ','a a','o d','os ','as ','no ','na ',
    ' en','ent','ant','men','nte','est','ara','que','par','pra',
    'ade','ida','ada','udo','sta','ndo','ura','ção','ões','aria',
    'or ','e d','s d','a d','o a','a e','e e','a o','o o',
    ' o ',' e ',' a ',' s ',' n ',' m ',' d ',' t ',' r ',' l ',
]
_NGRAMAS_3 = list(dict.fromkeys(_NGRAMAS_BASE))[:40]
_NGRAMAS_4 = [ng[0:4] for ng in ['ment','ação','ções','dade','tudo','para','ente','ndo','ando','aria','ível','ável','ória','ório','ário','íssimo','adamente','mente ','dade','port','rodu','trabal','scien','investi','estud','metod','resul','discuss','concord','explíc','implíc','signif','tific','caract','aplic','const','educ','proced','exper','muito','sempr','açõe','trab','ncia','acad','super']]
_NGRAMAS_4 = [ng for ng in _NGRAMAS_4 if len(ng) == 4]
_NGRAMAS = list(dict.fromkeys(_NGRAMAS_3 + _NGRAMAS_4))[:60]

# === Famílias de características (para Figura 1) ===
FAMILIAS_FEATURES = {
    'Lexicais': [
        'ttr_raw', 'guiraud_r', 'honore_hapax', 'simpson_d', 'comprimento_medio_palavras',
        'hapax_legomena_ratio', 'dis_legomena_ratio',
    ] + [f'freq_funcional_{w}' for w in PALAVRAS_FUNCIONAIS_PT],
    'Sintáticas': [
        'profundidade_mediana_dep', 'profundidade_media_dep',
        'razao_subordinadas_coordenadas', 'n_medio_filhos_token',
        'freq_dep_amod', 'freq_dep_advmod', 'freq_dep_nsubj',
        'freq_dep_obj', 'freq_dep_obl', 'freq_dep_cc', 'freq_dep_mark',
        'ratio_svs_per_frase', 'n_medio_verb_frase', 'n_medio_noun_frase',
        'n_medio_adj_frase', 'n_medio_adv_frase', 'n_medio_propn_frase', 'n_medio_adp_frase',
    ],
    'Estruturais': [
        'comprimento_medio_frases_palavras',
        'n_medio_frases_paragrafo',
        'n_paragrafos_por_1000_palavras',
        'freq_rel_ponto_virgula', 'freq_rel_virgulas', 'freq_rel_dois_pontos',
        'freq_rel_tracos', 'freq_rel_pontos_interrogacao',
        'freq_rel_pontos_exclamacao', 'freq_rel_reticencias',
        'freq_rel_parenteses_curvos', 'freq_rel_aspas',
        'n_medio_palavras_paragrafo',
        'n_exclamacao_por_100frases', 'n_interrogacao_por_100frases',
        'aspas_por_100frases',
    ],
    'N-gramas_caracteres': [f'freq_char_{ng}' for ng in _NGRAMAS],
}


# === Modelo de classificação (§3.4: SVM com kernel RBF foi o selecionado) ===
CLASSIFICADOR_PADRAO = 'svm_rbf'
HIPERPARAMETROS_SVM = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf'],
    'class_weight': ['balanced', None],
    'random_state': [SEED],
}
# Outros classificadores testados no artigo (§3.4): Naive Bayes, Random Forest, XGBoost
CLASSIFICADORES_TESTADOS = ['naive_bayes','random_forest','xgb','svm_rbf']

# === Protocolo experimental ===
FOLDS_EXTERNOS_LOOCV = 5  # validação cruzada aninhada (§3.4)
FOLDS_INTERNOS = 3
METRICA_OTIMIZACAO = 'f1_weighted'

# === Paleta de cores para gráficos
COR_BASELINE = '#999999'  # cinzento (trabalho único, tracejada Fig2)
COR_LONGITUDINAL = '#1f77b4'  # azul (longitudinal, contínua Fig2)
CORES_FAMILIAS = {
    'Lexicais': '#2E86AB',
    'Sintáticas': '#A23B72',
    'Estruturais': '#F18F01',
    'N-gramas_caracteres': '#C73E1D',
}

# === Impressão digital estendida
DESATIVAR_STANZA_PARSE = True  # pôr True para testes rápidos sem parser sintático Stanza
