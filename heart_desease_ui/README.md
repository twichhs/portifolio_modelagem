# Previsao de Risco de Doencas Cardiovasculares

Este projeto apresenta uma aplicacao em Streamlit para estimar risco de doenca cardiaca a partir de dados clinicos, sinais vitais e habitos de vida. O modelo utilizado e uma Regressao Logistica empacotada em um pipeline de pre-processamento e classificacao.

O objetivo principal e demonstrar uma etapa de colocacao de modelo em producao por meio de uma interface simples, acessivel e reutilizavel.

## Aviso Importante

Este projeto tem finalidade academica e educacional. A previsao gerada pelo modelo nao substitui avaliacao medica, diagnostico, tratamento ou orientacao profissional. Qualquer decisao relacionada a saude deve ser tomada com acompanhamento de um profissional qualificado.

## Funcionalidades

- Interface web em Streamlit para preenchimento dos dados do paciente.
- Formulario organizado por grupos de informacoes:
  - dados do paciente;
  - sinais vitais e exames;
  - informacoes cardiologicas;
  - estilo de vida.
- Previsao de classificacao de risco cardiaco.
- Exibicao da probabilidade estimada pelo modelo, quando disponivel.
- Visualizacao dos dados enviados ao modelo.
- Aba com metricas e graficos gerados durante a etapa de treino.

## Estrutura do Projeto

```text
.
|-- README.md
|-- model
|   |-- heart_disease_risk_dataset.parquet
|   |-- pipeline.pkl
|   |-- regressao_logistica_desafio.ipynb
|   |-- ui.py
|   `-- plots
|       |-- ap.png
|       |-- correlacao.png
|       |-- importancia_features_target.png
|       `-- roc_auc.png
|-- stream_lit_tutorial
`-- ui
```

## Principais Arquivos

- `model/ui.py`: aplicacao Streamlit usada para preencher o formulario e gerar previsoes.
- `model/pipeline.pkl`: pipeline treinado contendo pre-processamento e modelo de classificacao.
- `model/heart_disease_risk_dataset.parquet`: dataset utilizado no projeto.
- `model/regressao_logistica_desafio.ipynb`: notebook com a etapa de exploracao, treino e avaliacao.
- `model/plots/`: graficos exibidos na aba de metricas da interface.

## Dados de Entrada

O formulario coleta variaveis como:

- idade;
- sexo;
- pressao arterial sistolica e diastolica;
- colesterol total, HDL, LDL e triglicerideos;
- glicemia em jejum;
- hemoglobina glicada;
- IMC;
- frequencia cardiaca;
- tipo de dor no peito;
- depressao do segmento ST;
- historico familiar;
- historico de tabagismo;
- consumo de alcool;
- exercicio semanal;
- sono, estresse, passos diarios e qualidade da dieta.

## Como Executar

1. Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instale as dependencias principais:

```bash
pip install streamlit pandas scikit-learn joblib pyarrow
```

3. Execute a aplicacao:

```bash
streamlit run model/ui.py
```

4. Acesse a URL exibida pelo Streamlit no terminal.

## Modelo

O modelo salvo em `model/pipeline.pkl` e um pipeline do scikit-learn. Ele recebe os dados em formato tabular, aplica as transformacoes necessarias e retorna a classificacao prevista.

A aplicacao tambem tenta exibir a probabilidade estimada de risco usando `predict_proba`, caso esse metodo esteja disponivel no pipeline carregado.

## Metricas e Visualizacoes

A interface apresenta uma aba com:

- amostra do dataset original;
- matriz de correlacao;
- curva Precision-Recall;
- curva ROC;
- importancia das features.

Esses arquivos estao armazenados em `model/plots/`.

## Observacoes de Ambiente

O arquivo `pipeline.pkl` deve ser carregado preferencialmente com a mesma versao do scikit-learn usada no treinamento. Caso haja diferenca de versao, o Python pode emitir avisos de compatibilidade ao carregar o modelo.

Para maior reprodutibilidade, recomenda-se registrar as versoes das bibliotecas usadas no treino e na execucao da aplicacao.

## Fonte do Dataset

Dataset utilizado no treinamento:

https://www.kaggle.com/datasets/srisyra02/heart-disease-risk-2026
