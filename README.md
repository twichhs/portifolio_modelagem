# Portfólio de Modelagem e Ciência de Dados

Portfólio de projetos aplicados de Machine Learning, com foco em modelagem preditiva para problemas de classificação. Cada projeto documenta o raciocínio por trás da modelagem — hipóteses testadas, decisões de pré-processamento e leitura crítica dos resultados — e não apenas o código final.

<p align="left">
  <img src="https://img.shields.io/badge/Python-0A0A0A?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-0A0A0A?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-0A0A0A?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Streamlit-0A0A0A?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Matplotlib-0A0A0A?style=flat-square&logo=python&logoColor=white" alt="Matplotlib">
</p>

## Projetos

| Projeto | Problema | Modelo final | Métrica principal |
|---|---|---|---|
| [Detecção de Fraude em Cartão de Crédito](./credit_fraud_detection) | Classificação binária, classes desbalanceadas (1,7% fraude) | Regressão Logística com threshold ajustado | PR-AUC 0.35 |
| [Risco de Doença Cardiovascular](./heart_desease_ui) | Classificação binária | Regressão Logística com features padronizadas | F1 0.84 na classe de risco |

---

## 1. Detecção de Fraude em Transações de Cartão de Crédito

**Objetivo.** Treinar um classificador capaz de identificar transações fraudulentas em um dataset com forte desbalanceamento de classes: apenas 339 fraudes em 20.000 transações (1,7%).

**Abordagem.** Comparação de quatro modelos (Regressão Logística, Decision Tree, KNN e Random Forest) com validação cruzada estratificada de 5 folds e `class_weight='balanced'`. O diagnóstico mostrou que a limitação não estava na capacidade dos modelos de separar as classes — o Random Forest atingiu ROC-AUC de 0.91 — mas no threshold padrão de 0.5, inadequado para uma classe tão rara. O ponto de corte foi recalibrado a partir da curva precision-recall, maximizando o F1 da Regressão Logística, o modelo com melhor PR-AUC entre os quatro avaliados.

**Resultado.** PR-AUC de 0.35 no conjunto de teste, com precisão entre 0.55 e 0.58 e recall entre 0.24 e 0.49 na classe de fraude. Os sinais mais relevantes identificados foram `merchant_risk_score`, `velocity_score`, `cvv_retry_count` e ausência de autenticação na transação.

**Stack:** Python, Pandas, Scikit-Learn, Matplotlib, Seaborn.
**Projeto completo:** [`credit_fraud_detection/`](./credit_fraud_detection)

---

## 2. Previsão de Risco de Doença Cardiovascular

**Objetivo.** Treinar um modelo de Regressão Logística para estimar o risco de doença cardíaca a partir de dados clínicos, sinais vitais e hábitos de vida, e disponibilizá-lo em uma interface interativa.

**Abordagem.** Comparação entre features originais e padronizadas com `StandardScaler`, validação cruzada com K-Fold (5 splits) e empacotamento do pipeline completo — pré-processamento e modelo — em um único `Pipeline` do scikit-learn pronto para produção.

**Resultado.** Padronizar as features elevou a acurácia de 0.83 para 0.90 e o F1 da classe de risco de 0.70 para 0.84 (precisão 0.86, recall 0.82). O modelo final foi disponibilizado em uma aplicação Streamlit onde o usuário preenche um formulário clínico e recebe a previsão de risco junto com a probabilidade estimada.

**Stack:** Python, Scikit-Learn, Streamlit, Pandas.
**Projeto completo:** [`heart_desease_ui/`](./heart_desease_ui)

---

## Contato

[LinkedIn](https://br.linkedin.com/in/leonardo-barros-07330822a)
