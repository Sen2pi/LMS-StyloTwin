# LMS-StyloTwin: Longitudinal Stylometric Profiling

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![spaCy](https://img.shields.io/badge/spaCy-NLP-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Uma arquitetura computacional desenhada para auditoria de integridade académica em ambientes de *e-learning* (como o Moodle). O **LMS-StyloTwin** ultrapassa as abordagens tradicionais de deteção binária (Humano vs. IA), estabelecendo perfis estilométricos longitudinais por estudante para captar a evolução natural da escrita e reduzir a taxa de falsos positivos na deteção de coautoria homem-máquina.

## 🚀 Principais Funcionalidades

* **Extração Multidimensional:** Processamento de linguagem natural (via `spaCy`) para extração de dezenas de métricas léxicas, de caráter, sintáticas e estruturais.
* **Construção de Perfil Longitudinal:** Agregação de histórico de submissões usando vetores centróides com cálculo de distância (Manhattan ponderada pela variância).
* **Prevenção de *Data Leakage*:** *Pipelines* rígidas no `scikit-learn` que garantem que a normalização (Z-score), imputação e redução de dimensionalidade (TF-IDF + PCA) ocorrem estritamente no conjunto de treino.
* **Classificação de Granularidade Fina:** Deteção de texto 100% humano, assistido por IA, híbrido (50/50) e 100% gerado por Grandes Modelos de Linguagem (LLMs).
* **Anonimização *By-Design*:** Estrutura preparada para receber chaves de pseudoanonimização, garantindo total conformidade com o RGPD durante as análises de *Learning Analytics*.

## 🏗️ Arquitetura do Repositório

```text
├── data/
│   ├── raw/                 # Dados originais anonimizados (não versionados)
│   ├── processed/           # Textos limpos (remoção de citações, tabelas, etc.)
│   └── external/            # Datasets de validação (ex: PAN-2020 Cross-Domain)
├── notebooks/               # Jupyter Notebooks para análise exploratória e estatística (ICC)
├── src/
│   ├── preprocessing.py     # Limpeza de ruído académico e segmentação
│   ├── feature_extraction.py# Extração de características estilométricas (spaCy)
│   ├── model_pipeline.py    # Construção da pipeline scikit-learn (PCA, VIF, Normalização)
│   └── evaluation.py        # Validação aninhada e extração da matriz de confusão
├── config.yaml              # Definição de seeds (ex: 42) e parâmetros de execução
├── requirements.txt         # Dependências do projeto
└── README.md
