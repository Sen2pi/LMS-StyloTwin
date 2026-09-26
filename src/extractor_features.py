"""
extractor_features.py
=====================
Extrai 204 características estilométricas em 4 famílias (conforme Artigo V2):
  - Léxicas (TTR, riqueza vocabular, frequências de 80 palavras funcionais, comprimento palavras, hapax)
  - Sintáticas (árvore de dependências, tipos de relação, POS tags, razão subordinação/coordenação)
  - Estruturais (pontuação, frases por parágrafo, comprimento de frases, etc.)
  - N-gramas de caracteres (40 3-gramas + 20 4-gramas mais frequentes em PT)
Total: 204 features exatas do artigo.
"""

import os
import sys
import re
import string
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / '_cache'
_MPL = _CACHE / 'matplotlib'
_NLTKDATA = _CACHE / 'nltk_data'
_MPL.mkdir(parents=True, exist_ok=True)
_NLTKDATA.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(_MPL)
os.environ['NLTK_DATA'] = str(_NLTKDATA)
os.environ['NUMEXPR_MAX_THREADS'] = os.environ.get('NUMEXPR_MAX_THREADS', '8')

_VENDOR = _ROOT / '_vendor'
if _VENDOR.exists() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from tqdm.auto import tqdm

from .config import (
    PALAVRAS_FUNCIONAIS_PT,
    _NGRAMAS,
    DESATIVAR_STANZA_PARSE,
    SEED,
)

warnings.filterwarnings('ignore')

# === Inicialização de ferramentas NLTK ===
_NLTK_DOWNLOAD_DIR = str(_NLTKDATA)

def _ensure_nltk():
    global _NLTK_DOWNLOAD_DIR
    if _NLTK_DOWNLOAD_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DOWNLOAD_DIR)
    for pkg in ['punkt','stopwords','punkt_tab','rslp','averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}' if pkg=='stopwords' else pkg)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True, download_dir=_NLTK_DOWNLOAD_DIR)
            except Exception:
                    try:
                        nltk.download(pkg, quiet=True)
                    except Exception:
                        pass

_ensure_nltk()
from nltk.corpus import stopwords
STOPWORDS_PT = set(stopwords.words('portuguese'))

# === Tokenizadores ===
TOKENIZADOR_PALAVRAS = RegexpTokenizer(r"[A-Za-zÀ-ÿçÇâêîôûàáéíóúãõÃÕ\-']+")
TOKENIZADOR_FRASES = sent_tokenize

# === Inicialização Stanza para parsing sintático ===
_STANZA_NLP = None

def _inicializar_stanza():
    global _STANZA_NLP
    if DESATIVAR_STANZA_PARSE or _STANZA_NLP is not None:
        return _STANZA_NLP
    try:
        import stanza
    except ImportError:
        warnings.warn("Stanza não instalado; usar fallback heurístico para features sintáticas. "
                      "Para features sintáticas exatas: pip install stanza, depois DESATIVAR_STANZA_PARSE=False.")
        _STANZA_NLP = None
        return _STANZA_NLP
    try:
        stanza.download('pt', verbose=False)
    except Exception:
        pass
    # Modelo leve, tokenize+mwt+pos+lemma+depparse para PT
    try:
        _STANZA_NLP = stanza.Pipeline(
            lang='pt',
            processors='tokenize,mwt,pos,lemma,depparse',
            tokenize_no_ssplit=True,
            verbose=False,
            use_gpu=False,
        )
    except Exception:
        try:
            _STANZA_NLP = stanza.Pipeline(lang='pt', verbose=False, use_gpu=False)
        except Exception:
            warnings.warn("Stanza inicialização falhou; usar heurísticas sintáticas.")
            _STANZA_NLP = None
    return _STANZA_NLP

# ====================================================================
# Funções auxiliares de pré-processamento
# ====================================================================
def _limpar_texto(texto: str) -> str:
    return re.sub(r'\s+', ' ', texto.strip())

def _separar_paragrafos(texto: str) -> List[str]:
    return [p.strip() for p in re.split(r'\n\s*\n', texto) if p.strip()]

def _contar_pontuacao(texto: str) -> Counter:
    return Counter(c for c in texto if c in string.punctuation + '«»“”‘’–—…')

# ====================================================================
# 1) Características LÉXICAS (~87 features)
# ====================================================================
def _extrair_lexicas(texto_limpo: str, palavras: List[str], tokens_minusculos: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    n_palavras = max(1, len(palavras))
    n_tokens = len(tokens_minusculos) or 1
    palavras_sem_stop = [p for p in tokens_minusculos if p not in STOPWORDS_PT and len(p) > 2]
    n_palavras_sem_stop = max(1, len(palavras_sem_stop))
    freq = Counter(tokens_minusculos)
    tipos = set(tokens_minusculos)
    n_tipos = len(tipos)

    # Riqueza vocabular (7 features)
    feats['ttr_raw'] = n_tipos / n_tokens  # Type-Token Ratio bruto
    feats['guiraud_r'] = n_tipos / np.sqrt(n_tokens)  # Índice de Guiraud (§4.1, ICC=0,66)
    # Honoré Hapax: Hapax / N_tokens * 100
    hapax = [w for w, c in freq.items() if c == 1]
    dis = [w for w, c in freq.items() if c == 2]
    feats['hapax_legomena_ratio'] = len(hapax) / n_tokens
    feats['dis_legomena_ratio'] = len(dis) / n_tokens
    feats['honore_hapax'] = (100 * np.log(n_tokens)) / max(1e-9, 1 - (len(hapax) / n_tipos)) if n_tipos>0 else 0.0
    feats['simpson_d'] = sum(c*(c-1) for c in freq.values()) / (n_tokens * (n_tokens-1)) if n_tokens>1 else 0.0
    feats['comprimento_medio_palavras'] = np.mean([len(w) for w in palavras]) if palavras else 0.0  # §4.1, ICC=0,58

    # Frequências relativas de palavras funcionais (80 features)
    for pal in PALAVRAS_FUNCIONAIS_PT:
        feats[f'freq_funcional_{pal}'] = freq.get(pal.lower(), 0) / n_tokens

    return feats

# ====================================================================
# 2) Características SINTÁTICAS (17 features, com ou sem Stanza)
# ====================================================================
def _extrair_sintaticas(frases: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    n_frases = max(1, len(frases))
    nlp = _inicializar_stanza()

    profundidades: List[int] = []
    n_filhos_total: List[int] = []
    contagem_deps: Counter = Counter()
    total_tokens_sintaticos = 0
    n_subordinadas: int = 0
    n_coordenadas: int = 0
    contagem_pos: Counter = Counter()

    if nlp is not None and not DESATIVAR_STANZA_PARSE:
        try:
            import stanza
            doc = nlp('\n'.join(frases))
            for sentenca in doc.sentences:
                # profundidade da árvore
                ids_head = {}
                for tok in sentenca.words:
                    ids_head[tok.id] = tok.head
                # calcular profundidade
                def _profundidade(id_tok, memo=None):
                    memo = memo if memo is not None else {}
                    if id_tok in memo:
                        return memo[id_tok]
                    if id_tok == 0 or ids_head.get(id_tok, 0) == 0:
                        memo[id_tok] = 1
                        return 1
                    p = 1 + _profundidade(ids_head[id_tok], memo)
                    memo[id_tok] = p
                    return p
                depths_sent = [_profundidade(t.id) for t in sentenca.words]
                if depths_sent:
                    profundidades.append(np.median(depths_sent))
                    n_filhos_total.append(np.mean(list(Counter(ids_head.values()).values()) or [1]))
                total_tokens_sintaticos += len(sentenca.words)
                for tok in sentenca.words:
                    deprel = (tok.deprel or '').lower()
                    contagem_deps[deprel.split(':')[0]] += 1
                    contagem_pos[(tok.upos or 'X').lower()] += 1
                    # subordinação vs coordenação
                    if deprel in ('mark','acl','advcl','ccomp','xcomp','csubj','acl:relcl'):
                        n_subordinadas += 1
                    elif deprel in ('cc','conj'):
                        n_coordenadas += 1
        except Exception as e:
            print(f"[WARN] Stanza falhou: {e}. A usar heurísticas fallback para sintáticas.")

    # Fallback se Stanza não estiver disponível: heurísticas POS com NLTK simples e marcadores
    if not profundidades:
        # Estimativas sintáticas por heurística (usadas para testes rápidos)
        marcadores_sub = ['que','quando','enquanto','porque','pois','portanto','se','bem como','de modo que','a fim de','embora','se bem que','já que','uma vez que']
        marcadores_cord = ['e','ou','mas','porém','todavia','contudo','não só','como também','ora','quer']
        for frase in frases:
            toks = frase.lower().split()
            frase_original = frase
            frase_low = frase.lower()
            n_subordinadas += sum(1 for m in marcadores_sub if m in frase_low.split() or m in frase_low)
            n_coordenadas += sum(1 for m in marcadores_cord if m in frase_low.split() or m in frase_low)
            # estimativa comprimento de "árvore" por nº de palavras por frase
            profundidades.append(min(10, 1 + len(toks) // 8))
            # heurística nº médio filhos: baseado na segmentação por vírgulas e conjunções
            segmentos = max(1, 1 + frase.count(',') + frase.count(';') + frase.count(' e ') + frase.count(' ou '))
            n_filhos_total.append(max(1, len(toks) / segmentos))
            # POS dummy (contagem de adjetivos/adverbios por sufixos típicos PT)
            for w in frase.split():
                w_punc = w.strip(string.punctuation + '«»“”‘’–—…')
                wl = w_punc.lower()
                is_maiuscula = len(w_punc) > 0 and w_punc[0].isupper()
                if (wl.endswith('o') or wl.endswith('a') or wl.endswith('os') or wl.endswith('as')
                        or wl.endswith('ês') or wl.endswith('esa')):
                    if len(wl) > 3 and wl not in STOPWORDS_PT:
                        contagem_pos['adj'] += 1
                if wl.endswith('mente'):
                    contagem_pos['adv'] += 1
                if (wl.endswith('r') or wl.endswith('s') or wl.endswith('m') or wl.endswith('v')
                        or wl.endswith('z') or wl.endswith('ndo') or wl.endswith('ado')
                        or wl.endswith('ido') or wl.endswith('ste') or wl.endswith('ámos')):
                    if len(wl) > 2 and wl not in STOPWORDS_PT:
                        contagem_pos['verb'] += 1
                if len(w_punc) > 3 and is_maiuscula and wl not in STOPWORDS_PT:
                    contagem_pos['propn'] += 1
                if wl in ('de','em','para','por','com','sem','sobre','entre','através','durante'):
                    contagem_pos['adp'] += 1
                if (wl not in STOPWORDS_PT and len(wl) > 3 and not
                        (wl.endswith('mente'))):
                    contagem_pos['noun'] += 1
            total_tokens_sintaticos += len(toks)

    # features sintáticas finais
    total_deps = max(1, total_tokens_sintaticos)
    feats['profundidade_mediana_dep'] = float(np.median(profundidades)) if profundidades else 0.0  # §4.1 ICC=0,76
    feats['profundidade_media_dep'] = float(np.mean(profundidades)) if profundidades else 0.0
    feats['n_medio_filhos_token'] = float(np.mean(n_filhos_total)) if n_filhos_total else 0.0  # §4.1 ICC=0,69
    feats['razao_subordinadas_coordenadas'] = (n_subordinadas / max(1, n_coordenadas))  # §4.1 ICC=0,74
    feats['ratio_svs_per_frase'] = (n_subordinadas + n_coordenadas) / n_frases

    # frequências de relações de dependência (10 features)
    deps_alvo = ['amod','advmod','nsubj','obj','obl','cc','mark','det','case','aux']
    for d in deps_alvo:
        feats[f'freq_dep_{d}'] = contagem_deps.get(d, 0) / total_deps

    # POS médias por frase (6 features)
    for pos in ['verb','noun','adj','adv','propn','adp']:
        feats[f'n_medio_{pos}_frase'] = contagem_pos.get(pos, 0) / n_frases

    return feats

# ====================================================================
# 3) Características ESTRUTURAIS (15 features)
# ====================================================================
def _extrair_estruturais(texto: str, frases: List[str], palavras: List[str], paragrafos: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    n_frases = max(1, len(frases))
    n_palavras = max(1, len(palavras))
    n_paragrafos = max(1, len(paragrafos))
    pont = _contar_pontuacao(texto)

    # Comprimento de frases e parágrafos
    feats['comprimento_medio_frases_palavras'] = np.mean([len(word_tokenize(f, language='portuguese')) for f in frases]) if frases else 0.0  # §4.1 ICC=0,64
    feats['n_medio_frases_paragrafo'] = n_frases / n_paragrafos  # §4.1 ICC=0,61
    feats['n_medio_palavras_paragrafo'] = n_palavras / n_paragrafos
    feats['n_paragrafos_por_1000_palavras'] = n_paragrafos / (n_palavras / 1000)

    # Frequências relativas de pontuação (9 features)
    feats['freq_rel_ponto_virgula'] = pont.get(';', 0) / n_palavras * 1000  # §4.1 ICC=0,68
    feats['freq_rel_virgulas'] = pont.get(',', 0) / n_palavras * 1000
    feats['freq_rel_dois_pontos'] = pont.get(':', 0) / n_palavras * 1000
    feats['freq_rel_tracos'] = sum(pont.get(c,0) for c in ['-','–','—']) / n_palavras * 1000
    feats['freq_rel_pontos_interrogacao'] = pont.get('?', 0) / n_palavras * 1000
    feats['freq_rel_pontos_exclamacao'] = pont.get('!', 0) / n_palavras * 1000
    feats['freq_rel_reticencias'] = pont.get('…', 0) + pont.get('.', 0) // 3 / n_palavras * 1000
    feats['freq_rel_parenteses_curvos'] = (pont.get('(', 0) + pont.get(')', 0)) / n_palavras * 1000  # §4.1 ICC=0,57
    feats['freq_rel_aspas'] = sum(pont.get(c, 0) for c in ['"',"'",'«','»','“','”','‘','’']) / n_palavras * 1000

    # Extras de organização
    feats['n_exclamacao_por_100frases'] = pont.get('!', 0) / n_frases * 100
    feats['n_interrogacao_por_100frases'] = pont.get('?', 0) / n_frases * 100
    feats['aspas_por_100frases'] = sum(pont.get(c, 0) for c in ['"',"'",'«','»','“','”','‘','’']) / n_frases * 100

    return feats

# ====================================================================
# 4) N-gramas de caracteres (60 features: 40 3-gramas + 20 4-gramas)
# ====================================================================
def _extrair_ngramas_caracteres(texto: str, n=3) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    texto_l = texto.lower()
    total_ngrams: Counter = Counter()
    for i in range(len(texto_l) - n + 1):
        ng = texto_l[i:i+n]
        if all(c.isalpha() or c.isspace() for c in ng):
            total_ngrams[ng] += 1
    total = max(1, sum(total_ngrams.values()))
    return total_ngrams, total

def _extrair_ngramas_features(texto: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    ngramas_counts_3, total3 = _extrair_ngramas_caracteres(texto, n=3)
    ngramas_counts_4, total4 = _extrair_ngramas_caracteres(texto, n=4)
    for ng in _NGRAMAS:
        nlen = len(ng.strip())
        if nlen == 3:
            feats[f'freq_char_{ng}'] = ngramas_counts_3.get(ng, 0) / total3
        elif nlen == 4:
            feats[f'freq_char_{ng}'] = ngramas_counts_4.get(ng, 0) / total4
        else:
            feats[f'freq_char_{ng}'] = 0.0
    return feats

# ====================================================================
# Função principal de extração (chama todas acima)
# ====================================================================
def extrair_features_texto(texto: str, id_texto: Optional[str] = None) -> Dict[str, float]:
    texto = _limpar_texto(texto)
    paragrafos = _separar_paragrafos(texto)
    frases = TOKENIZADOR_FRASES(texto, language='portuguese')
    palavras = TOKENIZADOR_PALAVRAS.tokenize(texto)
    tokens_minusculos = [p.lower() for p in palavras if p.strip()]

    feats: Dict[str, float] = {}
    feats.update(_extrair_lexicas(texto, palavras, tokens_minusculos))
    feats.update(_extrair_sintaticas(frases))
    feats.update(_extrair_estruturais(texto, frases, palavras, paragrafos))
    feats.update(_extrair_ngramas_features(texto))

    if id_texto is not None:
        feats['id_texto'] = id_texto
    return feats


def extrair_features_corpus(
    lista_textos: List[Dict[str, str]],
    coluna_texto: str = 'texto',
    barra_progresso: bool = True,
) -> pd.DataFrame:
    """
    Recebe lista de dicionários: [{'texto':..., 'id_texto':'S1_T1', 'estudante':'S1', ...}]
    Retorna DataFrame com 204 features + colunas meta (id_texto, estudante, etc.)
    """
    resultados: List[Dict[str, float]] = []
    iter_lista = tqdm(lista_textos, desc='Extraindo features', disable=not barra_progresso)
    for item in iter_lista:
        texto = item[coluna_texto]
        id_texto = item.get('id_texto')
        feats = extrair_features_texto(texto, id_texto=id_texto)
        # Incluir metadados
        for k, v in item.items():
            if k not in feats:
                feats[k] = v
        resultados.append(feats)

    df = pd.DataFrame(resultados)
    # Garantir ordem e que não temos NaN
    df = df.fillna(0.0)
    return df


# === Nome das 204 features ===
def lista_todas_features() -> List[str]:
    from .config import FAMILIAS_FEATURES
    todas = []
    for fam in FAMILIAS_FEATURES.values():
        for f in fam:
            if f not in todas:
                todas.append(f)
    return todas
