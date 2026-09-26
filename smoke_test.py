import os, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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

print("SMOKE TEST 1: Importar dependências...", flush=True)
try:
    import numpy as np
    print(f"  OK numpy {np.__version__}", flush=True)
except Exception as e:
    print(f"  FALHA numpy: {type(e).__name__}: {e}", flush=True); sys.exit(1)

try:
    import pandas as pd
    print(f"  OK pandas {pd.__version__}", flush=True)
except Exception as e:
    print(f"  FALHA pandas: {type(e).__name__}: {e}", flush=True); sys.exit(1)

try:
    import scipy
    print(f"  OK scipy {scipy.__version__}", flush=True)
except Exception as e:
    print(f"  FALHA scipy: {type(e).__name__}: {e}", flush=True); sys.exit(1)

try:
    import sklearn
    print(f"  OK sklearn {sklearn.__version__}", flush=True)
except Exception as e:
    print(f"  FALHA sklearn: {type(e).__name__}: {e}", flush=True); sys.exit(1)

try:
    import pingouin as pg
    print(f"  OK pingouin {pg.__version__}", flush=True)
except Exception as e:
    print(f"  FALHA pingouin: {type(e).__name__}: {e}", flush=True); sys.exit(1)

print("\nSMOKE TEST 2: Importar módulos do projeto...", flush=True)
try:
    import src.config as cfg
    print(f"  OK config (SEED={cfg.SEED}, N_FAMILIAS={len(cfg.FAMILIAS_FEATURES)})", flush=True)
except Exception as e:
    print(f"  FALHA config: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    from src import corpus_sintetico
    print(f"  OK corpus_sintetico importado", flush=True)
except Exception as e:
    print(f"  FALHA corpus_sintetico: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc(); sys.exit(1)

print("\nSMOKE TEST 3: Gerar corpus pequeno (5 estudantes, 2-3 submissões)...", flush=True)
try:
    lista_meta, df_meta = corpus_sintetico.gerar_corpus_sintetico(
        n_estudantes=5, n_submissoes_min=2, n_submissoes_max=3,
        guardar_ficheiros_txt=False, seed=42
    )
    print(f"  OK — gerados {len(lista_meta)} textos, {len(df_meta)} registos meta", flush=True)
    print(f"     Primeiro texto: estudante={lista_meta[0]['estudante']}, autoria={lista_meta[0]['autoria']}, len={len(lista_meta[0]['texto'])} chars", flush=True)
except Exception as e:
    print(f"  FALHA gerar corpus: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc(); sys.exit(1)

print("\nSMOKE TEST 4: Extrair features do primeiro texto...", flush=True)
try:
    from src.extractor_features import extrair_features_texto
    texto_teste = lista_meta[0]['texto']
    id_teste = lista_meta[0].get('id_texto')
    feat = extrair_features_texto(texto_teste, id_texto=id_teste)
    print(f"  OK — extraídas {len(feat)} features", flush=True)
    ttr_val = feat.get('ttr_raw', 0) if isinstance(feat.get('ttr_raw', 0), (int, float)) else 0.0
    guir_val = feat.get('guiraud_r', 0) if isinstance(feat.get('guiraud_r', 0), (int, float)) else 0.0
    cmf_val = feat.get('comprimento_medio_frases_palavras', 0) if isinstance(feat.get('comprimento_medio_frases_palavras', 0), (int, float)) else 0.0
    print(f"     Exemplos: TTR={ttr_val:.4f}, Guiraud={guir_val:.4f}, CMF={cmf_val:.2f}", flush=True)
except Exception as e:
    print(f"  FALHA extrair features: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc(); sys.exit(1)

print("\nSMOKE TEST 5: Extrair features de todo o corpus...", flush=True)
try:
    from src.extractor_features import extrair_features_corpus
    df_feat = extrair_features_corpus(lista_meta, coluna_texto='texto', barra_progresso=False)
    print(f"  OK — shape={df_feat.shape} (linhas=textos, cols=meta+features)", flush=True)
    print(f"     Colunas meta visíveis: {[c for c in df_feat.columns[:10] if c in ['id_texto','estudante','submissao_idx','autoria']]}", flush=True)
except Exception as e:
    print(f"  FALHA extrair corpus features: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc(); sys.exit(1)

print("\n=================================")
print("🎉 SMOKE TEST SUCESSO — tudo pronto para o pipeline principal!")
