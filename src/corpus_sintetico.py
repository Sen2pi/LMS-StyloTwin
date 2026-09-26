"""
corpus_sintetico.py
===================
Gera um corpus sintético de textos académicos em PT para testar o pipeline.
O corpus imita os padrões observados no artigo:
  - M estudantes (20 a 50)
  - Cada estudante tem entre 3 e 11 submissões (média do artigo real: 11)
  - Para cada estudante, temos "estilos" próprios (frequências de palavras funcionais,
    comprimento de frases, profundidade sintática, etc.) com baixa variabilidade INTRA-estudante
    e alta variabilidade INTER-estudante → ICC elevado nas top features.
  - Inclui 4 categorias de autoria (conforme Quadro 4):
      1. humana (autêntica do estudante) → ~60% do corpus
      2. ia_assistida (humano revisou texto com IA, ligeiras alterações estilo) → ~15%
      3. hibrida (50% humano, 50% IA, alterações substanciais de estilo) → ~15%
      4. ia_gerada (texto integralmente por LLM, perfil estilo muito distinto) → ~10%
  - Os textos são sintéticos (parágrafos reais adaptados), mas com CONTAGENS E CARACTERÍSTICAS
    ESTILOMÉTRICAS ajustadas matematicamente para reproduzir os padrões do artigo.
"""

from typing import Dict, List, Tuple
import re
import os
import json
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (
    SEED,
    PALAVRAS_FUNCIONAIS_PT,
    CORPUS_SINTETICO_DIR,
)

warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)

# Modelo de texto base para sintetizar (frases académicas PT típicas)
FRASES_TIPICAS_ACADEMICAS_PT = [
    "O presente estudo teve como objetivo principal investigar a relação entre os perfis estilométricos longitudinais e a verificação de autoria no ensino superior.",
    "Nos últimos anos, a proliferação de ferramentas generativas baseadas em modelos de linguagem de grande dimensão transformou as práticas de escrita académica.",
    "A integridade académica constitui um pilar fundamental das instituições de ensino superior, especialmente em contextos de educação a distância.",
    "A metodologia adotada recorreu a um desenho longitudinal, com recolha de múltiplas submissões ao longo de dois semestres letivos consecutivos.",
    "Os dados foram recolhidos na plataforma Moodle da Universidade Aberta, com consentimento informado prévio de todos os participantes.",
    "A amostra final integrou textos académicos de natureza diversa: relatórios de atividade, reflexões críticas, ensaios argumentativos e análises de caso.",
    "As características estilométricas extraídas compreenderam quatro famílias distintas: léxicas, sintáticas, estruturais e n-gramas de caracteres.",
    "A estabilidade das características foi aferida mediante o cálculo do Coeficiente de Correlação Intraclasse, com limiares de estabilidade pré-definidos.",
    "Os resultados revelaram que as palavras funcionais, e nomeadamente as preposições e os artigos definidos, exibem elevada estabilidade intra-autor.",
    "Complementarmente, a profundidade mediana das árvores de dependências sintáticas e a razão entre estruturas subordinadas e coordenadas revelaram-se marcadores robustos de autoria.",
    "A comparação entre a abordagem de trabalho único e o perfil longitudinal mostrou uma redução estatisticamente significativa da taxa de falsos positivos.",
    "Esta melhoria no FPR não ocorreu ao preço de uma redução correspondente na revocação, que se manteve estatisticamente constante entre as duas condições experimentais.",
    "As implicações práticas para as instituições de ensino superior são relevantes, nomeadamente ao nível da justiça processual.",
    "Os sistemas de verificação de autoria deverão, no futuro, integrar um perfil longitudinal acumulado ao longo do percurso académico do estudante.",
    "A transparência na apresentação dos resultados, mediante dashboards interpretáveis para os docentes, assume particular importância.",
    "A presente investigação apresenta cinco limitações principais que devem ser consideradas na interpretação dos resultados.",
    "Futuras investigações deverão replicar o estudo com um corpus multicêntrico e explorar técnicas de interpretabilidade baseadas em valores SHAP e LIME.",
    "A atualização adaptativa do perfil longitudinal, ponderando mais fortemente os trabalhos recentes do que os mais antigos, constitui uma direção promissora.",
    "A evidência empírica reunida suporta a hipótese central da investigação: a informação longitudinal melhora a separabilidade das classes de autoria.",
    "Em suma, os perfis estilométricos longitudinais constituem uma ferramenta válida e fiável para a verificação de autoria em contexto de ensino superior português.",
    "Cada submissão foi anonimizada antes da extração de características, garantindo a conformidade com os princípios éticos da investigação com seres humanos.",
    "A normalização das variáveis de entrada foi efetuada mediante z-score, com parâmetros ajustados apenas ao conjunto de treino em cada fold da validação cruzada.",
    "O modelo final selecionado foi uma máquina de vetores de suporte com kernel de base radial, por apresentar melhor desempenho em validação cruzada aninhada.",
    "A calibração do limiar de decisão foi realizada para igualar a revocação entre as duas condições, conforme estabelecido no protocolo experimental pré-registado.",
    "As curvas ROC foram comparadas mediante o teste de DeLong, com nível de significância alfa fixado em 5%.",
]


def _gerar_perfil_estilo_estudante(id_estudante: int) -> Dict[str, float]:
    """
    Gera um vetor de "estilo idiossincrático" para cada estudante.
    Valores baixos/altos para as características que sabemos serem mais estáveis no artigo:
    frequências de palavras funcionais, comprimento de frases, profundidade sintática, etc.
    Retorna Dict[feature_name] = bias (multiplicador ou offset) que esse estudante tem.
    """
    perfil = {}
    # 1) Frequência de palavras funcionais (80): cada estudante tem preferências próprias
    # Selecionar 25 para variar bastante inter estudante
    n_func_variaveis = 25
    ids_var = rng.choice(len(PALAVRAS_FUNCIONAIS_PT), size=n_func_variaveis, replace=False)
    for i, pal in enumerate(PALAVRAS_FUNCIONAIS_PT):
        if i in ids_var:
            perfil[f'f_{pal}'] = float(rng.normal(loc=0, scale=0.35))
        else:
            perfil[f'f_{pal}'] = float(rng.normal(loc=0, scale=0.10))
    # 2) Características sintáticas
    perfil['profundidade_sintatica'] = float(rng.normal(loc=0, scale=0.50))  # ICC 0.76
    perfil['razao_subord_coord'] = float(rng.normal(loc=0, scale=0.45))  # ICC 0.74
    perfil['n_filhos_token'] = float(rng.normal(loc=0, scale=0.35))  # ICC 0.69
    perfil['freq_amod'] = float(rng.normal(loc=0, scale=0.30))
    perfil['freq_advmod'] = float(rng.normal(loc=0, scale=0.27))
    perfil['freq_nsubj'] = float(rng.normal(loc=0, scale=0.25))
    # 3) Léxicas de riqueza
    perfil['guiraud'] = float(rng.normal(loc=0, scale=0.30))  # ICC 0.66
    perfil['honore'] = float(rng.normal(loc=0, scale=0.25))  # ICC 0.62
    perfil['comp_med_pal'] = float(rng.normal(loc=0, scale=0.25))  # ICC 0.58
    # 4) Estruturais
    perfil['freq_pv'] = float(rng.normal(loc=0, scale=0.35))  # ponto e vírgula, ICC 0.68
    perfil['comp_med_frase'] = float(rng.normal(loc=0, scale=0.35))  # ICC 0.64
    perfil['n_frases_par'] = float(rng.normal(loc=0, scale=0.30))  # ICC 0.61
    perfil['freq_parenteses'] = float(rng.normal(loc=0, scale=0.28))  # ICC 0.57
    return perfil


def _aplicar_perfil_a_texto(frases_base: List[str], perfil_estudante: Dict[str, float],
                            n_paragrafos: int = 6, n_frases_por_paragrafo: int = 4,
                            perturbacao_intra: float = 0.08,
                            categoria_autoria: str = 'humana',
                            ) -> str:
    """
    Gera um texto, aplicando o perfil do estudante + categoria de autoria.
    Varia n_frases_por_paragrafo com o perfil do estudante (ICC 0.61).
    """
    # Amostrar frases
    frases_escolhidas = list(frases_base)
    rng.shuffle(frases_escolhidas)

    # Ajustar n.º de parágrafos e frases por parágrafo conforme perfil
    n_frases_pp = int(np.clip(n_frases_por_paragrafo + perfil_estudante.get('n_frases_par', 0) * 2.0, 3, 8))
    n_paragrafos_ajustado = int(np.clip(n_paragrafos + int(perfil_estudante.get('n_frases_par', 0) * 0.7), 3, 10))

    # Ajustar n-gramas (substituir palavras funcionais)
    substituicoes_funcionais = []
    for pal in PALAVRAS_FUNCIONAIS_PT:
        bias_p = perfil_estudante.get(f'f_{pal}', 0) + rng.normal(0, perturbacao_intra)
        # se bias > 0 → queremos mais ocorrências; se <0 → menos ocorrências
        substituicoes_funcionais.append((pal, bias_p))

    # Construir parágrafos
    paragrafos = []
    for _ in range(n_paragrafos_ajustado):
        n_frases = n_frases_pp + int(perturbacao_intra * 15 * rng.integers(-1, 2))
        n_frases = int(np.clip(n_frases, 2, 12))
        frase_par = []
        for _ in range(n_frases):
            if len(frases_escolhidas) == 0:
                frases_escolhidas = list(frases_base)
                rng.shuffle(frases_escolhidas)
            frase = frases_escolhidas.pop(0)
            # Aplicar substituições e ajustes de palavras funcionais
            frase_mod = frase
            palavras = frase_mod.split()
            novas = []
            for w in palavras:
                wl = re.sub(r'[^\w]', '', w).lower()
                # Aumentar/diminuir frequência conforme perfil
                for pal, bias in substituicoes_funcionais:
                    if wl == pal:
                        p_remover = 1 / (1 + np.exp(10 * bias)) * 0.28 if bias < 0 else 0.0
                        p_adicionar_extra = 1 / (1 + np.exp(-10 * bias)) * 0.22 if bias > 0 else 0.0
                        if rng.random() < p_remover:
                            continue  # remover esta ocorrência
                        if rng.random() < p_adicionar_extra:
                            novas.append(pal if not w[0].isupper() else pal.capitalize())
                novas.append(w)
            frase_mod = ' '.join(novas)
            # Ajustar comprimento de frase: comp_med_frase e profundidade
            bias_comp = perfil_estudante.get('comp_med_frase', 0) + rng.normal(0, perturbacao_intra)
            # Adicionar orações subordinadas se bias positivo, encurtar se negativo
            oracoes_extra = [
                ", o que é consistente com a literatura prévia existente sobre a matéria.",
                ", tal como ilustrado pelos resultados apresentados em estudos anteriores.",
                ", revelando uma tendência estatística que merece análise mais aprofundada.",
                ", permitindo identificar padrões de comportamento estilométrico idiossincrático.",
                ", dado o contexto específico de educação a distância em ambiente digital.",
            ]
            if bias_comp > 0.0 and rng.random() < min(0.9, bias_comp * 1.1):
                frase_mod += rng.choice(oracoes_extra)
            elif bias_comp < -0.1 and len(frase_mod) > 60:
                partes = re.split(r'(?<=[,.])\s+', frase_mod)
                if len(partes) >= 3:
                    frase_mod = ' '.join(partes[:-1]) + '.'
            frase_par.append(frase_mod.strip())
        paragrafos.append(' '.join(frase_par).strip())

    texto = '\n\n'.join(paragrafos)

    # Ajustes por CATEGORIA DE AUTORIA
    if categoria_autoria == 'ia_assistida':
        # IA ligeira: substituir 8-12% do texto por expressões mais "formais genéricas", pouco impacto
        substitutos = [
            ('O presente estudo', 'O estudo'),
            ('teve como objetivo principal investigar', 'visa investigar'),
            ('Nos últimos anos', 'Nos últimos tempos'),
            ('A metodologia adotada recorreu a', 'A metodologia usada recorreu a'),
            ('Os dados foram recolhidos', 'Os dados foram coletados'),
            ('Os resultados revelaram que', 'Os resultados mostram que'),
            ('A comparação entre a abordagem', 'A comparação entre a estratégia'),
            ('As implicações práticas para as instituições', 'As implicações para as instituições'),
            ('A transparência na apresentação dos resultados', 'A clareza na apresentação dos resultados'),
        ]
        for a, b in rng.choice(substitutos, size=min(8, len(substitutos)), replace=False):
            texto = texto.replace(a, b, 1)
    elif categoria_autoria == 'hibrida':
        # 50%: reescrever 25-40% das frases (substituí-las por versões mais simplificadas/diferentes)
        paragrafos_lst = texto.split('\n\n')
        for i in range(len(paragrafos_lst)):
            if rng.random() < 0.42:
                frases_p = re.split(r'(?<=[.!?])\s+', paragrafos_lst[i])
                for j in range(len(frases_p)):
                    if rng.random() < 0.55:
                        frases_p[j] = rng.choice([
                            "Este aspeto revelou-se particularmente relevante no contexto do estudo.",
                            "A análise comparativa permitiu obter conclusões relevantes para a comunidade académica.",
                            "Os padrões detetados são coerentes com o pressuposto teórico que enquadrou a investigação.",
                            "É possível identificar uma relação direta entre as variáveis em análise.",
                            "As implicações destes achados merecem discussão mais detalhada noutro âmbito.",
                            "Por outro lado, não se deve ignorar a influência exercida pelas variáveis de contexto.",
                        ])
                paragrafos_lst[i] = ' '.join(frases_p)
        texto = '\n\n'.join(paragrafos_lst)
    elif categoria_autoria == 'ia_gerada':
        # LLM: introduz frases características + reduz entropia lexical + padrão repetitivo
        frases_llm = [
            "Em primeiro lugar, é crucial reconhecer a complexidade inerente ao fenómeno em estudo.",
            "Além disso, os resultados obtidos demonstram uma correlação estatisticamente significativa.",
            "Por conseguinte, pode concluir-se que existem implicações relevantes para a prática pedagógica.",
            "Não obstante, é importante considerar as limitações metodológicas da presente investigação.",
            "Deste modo, a adoção de estratégias de verificação longitudinal constitui uma medida recomendada.",
            "Nesse sentido, as instituições educativas devem implementar políticas de integridade mais robustas.",
            "Simultaneamente, é fundamental assegurar a transparência em todos os processos de avaliação.",
            "Assim sendo, os dados recolhidos fornecem evidência empírica suficiente para suportar as hipóteses.",
        ]
        paragrafos_lst = texto.split('\n\n')
        novo = []
        for p in paragrafos_lst:
            frases_p = re.split(r'(?<=[.!?])\s+', p)
            # Substituir 60% por frases de IA e adicionar conectores
            novas = []
            for frase in frases_p:
                if rng.random() < 0.6:
                    novas.append(rng.choice(frases_llm))
                else:
                    # Encolher frases longas
                    if len(frase.split()) > 18:
                        novas.append(' '.join(frase.split()[:18]).rstrip(',') + '.')
                    else:
                        novas.append(frase)
            novo.append(' '.join(novas))
        texto = '\n\n'.join(novo)

    return texto.strip()


def gerar_corpus_sintetico(
    n_estudantes: int = 25,
    n_submissoes_min: int = 5,
    n_submissoes_max: int = 11,
    output_dir: str | os.PathLike | None = None,
    guardar_ficheiros_txt: bool = False,
    seed: int = SEED,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Gera o corpus sintético.
    :return: (lista_textos_meta, df_meta)
      lista_textos_meta = [{'id_texto': 'S01_T01','estudante':'S01','submissao_idx':1,'autoria':'humana','texto':...}]
    """
    global rng
    rng = np.random.default_rng(seed)
    out = output_dir or CORPUS_SINTETICO_DIR
    out = str(out)
    os.makedirs(out, exist_ok=True)

    lista: List[Dict] = []
    perfis_estudantes = {}
    for i in range(1, n_estudantes + 1):
        id_est = f'S{i:02d}'
        perfis_estudantes[id_est] = _gerar_perfil_estilo_estudante(i)
        n_sub = int(rng.integers(n_submissoes_min, n_submissoes_max + 1))
        for s_idx in range(1, n_sub + 1):
            # Definir categoria autoria: ~82% humana, 7% ia_assistida, 7% híbrida, 4% ia_gerada
            r = rng.random()
            if r < 0.82:
                cat = 'humana'
            elif r < 0.89:
                cat = 'ia_assistida'
            elif r < 0.96:
                cat = 'hibrida'
            else:
                cat = 'ia_gerada'
            texto = _aplicar_perfil_a_texto(
                frases_base=FRASES_TIPICAS_ACADEMICAS_PT,
                perfil_estudante=perfis_estudantes[id_est],
                n_paragrafos=int(rng.integers(4, 9)),
                n_frases_por_paragrafo=int(rng.integers(3, 6)),
                perturbacao_intra=0.09,  # Variação intra-estudante BAIXA
                categoria_autoria=cat,
            )
            id_texto = f'{id_est}_T{s_idx:02d}'
            item = {
                'id_texto': id_texto,
                'estudante': id_est,
                'submissao_idx': s_idx,
                'n_submissoes_total': n_sub,
                'autoria': cat,
                'texto': texto,
            }
            if guardar_ficheiros_txt:
                pasta_est = os.path.join(out, id_est)
                os.makedirs(pasta_est, exist_ok=True)
                caminho_txt = os.path.join(pasta_est, f'{id_texto}_{cat}.txt')
                with open(caminho_txt, 'w', encoding='utf-8') as f:
                    f.write(texto)
                item['caminho_arquivo'] = caminho_txt
            lista.append(item)
    df = pd.DataFrame(lista)
    # Guardar metadados JSON
    meta_path = os.path.join(out, 'metadados_corpus_sintetico.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_estudantes': n_estudantes,
            'n_textos_total': len(lista),
            'contagem_autoria': df['autoria'].value_counts().to_dict(),
            'contagem_submissoes': df.groupby('estudante')['submissao_idx'].max().to_dict(),
            'categorias_autoria': ['humana','ia_assistida','hibrida','ia_gerada'],
            'seed': seed,
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] Corpus sintético gerado: {len(lista)} textos, {df['estudante'].nunique()} estudantes, guardado em {out}")
    print(f"     Distribuição autoria: {df['autoria'].value_counts().to_dict()}")
    return lista, df
