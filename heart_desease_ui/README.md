# Previsão de Risco de Doença Cardiovascular

Aplicação em Streamlit para estimar o risco de doença cardíaca a partir de dados clínicos, sinais vitais e hábitos de vida. O modelo é uma Regressão Logística empacotada em um pipeline de pré-processamento e classificação, treinada para demonstrar a etapa de colocação de um modelo em produção por meio de uma interface simples e reutilizável.

**Dataset:** [Kaggle — Heart Disease Risk 2026](https://www.kaggle.com/datasets/srisyra02/heart-disease-risk-2026)

> Projeto com finalidade acadêmica e educacional. A previsão gerada pelo modelo não substitui avaliação médica, diagnóstico, tratamento ou orientação profissional.

## Metodologia

O modelo foi treinado e avaliado com validação cruzada K-Fold (5 splits). Um ponto central da modelagem foi comparar o desempenho com as features originais e com as features padronizadas por `StandardScaler`:

| Versão | Acurácia | Precisão (risco) | Recall (risco) | F1 (risco) |
|---|---|---|---|---|
| Features originais | 0.83 | 0.75 | 0.65 | 0.70 |
| Features padronizadas | 0.90 | 0.86 | 0.82 | 0.84 |

A padronização trouxe ganho consistente em todas as métricas, o que motivou o uso das features escaladas no modelo final. O pipeline de produção empacota pré-processamento e classificador em um único objeto scikit-learn, que recebe os dados brutos e devolve a previsão sem etapas manuais intermediárias.

## Funcionalidades da aplicação

A interface Streamlit é dividida em duas abas. Na aba de previsão, o usuário preenche um formulário organizado por grupos (dados do paciente, sinais vitais e exames, informações cardiológicas e estilo de vida) e recebe a classificação de risco junto com a probabilidade estimada, quando disponível pelo `predict_proba` do pipeline carregado. Na aba de métricas ficam os gráficos gerados durante o treino: amostra do dataset, matriz de correlação, curvas precision-recall e ROC, e importância das features.

As variáveis coletadas no formulário incluem idade, sexo, pressão arterial sistólica e diastólica, colesterol total, HDL, LDL e triglicerídeos, glicemia em jejum, hemoglobina glicada, IMC, frequência cardíaca, tipo de dor no peito, depressão do segmento ST, histórico familiar e de tabagismo, consumo de álcool, exercício semanal, sono, estresse, passos diários e qualidade da dieta.

## Estrutura

```text
heart_desease_ui/
├── README.md
├── ui.py
├── pipeline.pkl
├── heart_disease_risk_dataset.parquet
├── regressao_logistica_desafio.ipynb
├── requirements.txt
└── plots/
    ├── ap.png
    ├── correlacao.png
    ├── importancia_features_target.png
    └── roc_auc.png
```

`ui.py` é a aplicação Streamlit. `pipeline.pkl` contém o pipeline treinado de pré-processamento e classificação. `regressao_logistica_desafio.ipynb` documenta a etapa de exploração, treino e avaliação. `plots/` guarda os gráficos exibidos na aba de métricas.

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui.py
```

Acesse a URL exibida pelo Streamlit no terminal.

## Observações de ambiente

`pipeline.pkl` deve ser carregado preferencialmente com a mesma versão de scikit-learn usada no treino (`scikit-learn==1.7.2`, ver `requirements.txt`). Uma versão diferente pode gerar avisos de compatibilidade ao carregar o modelo.

## Stack

Python, Scikit-Learn, Streamlit, Pandas, PyArrow.
