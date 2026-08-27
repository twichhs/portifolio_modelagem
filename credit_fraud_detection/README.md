# Detecção de Fraude em Transações de Cartão de Crédito

Modelo de classificação binária para identificar transações fraudulentas, com foco em tratar corretamente um problema de classes fortemente desbalanceadas.

**Dataset:** [Kaggle — Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)

## O problema

O dataset tem 20.000 transações e 339 fraudes (1,7% do total), sem valores nulos. Nessa proporção, acurácia deixa de ser uma métrica confiável: um modelo que sempre prevê "não fraude" já acerta 98,3% dos casos. Por isso a avaliação foi feita com precision, recall, F1 e PR-AUC (average precision), mais informativa que ROC-AUC quando a classe positiva é rara.

## Metodologia

A base foi dividida em treino, validação e teste (60/20/20) com estratificação, necessária para preservar a proporção de fraude em cada conjunto dado o número reduzido de casos positivos. Quatro modelos foram comparados por validação cruzada estratificada (5 folds), todos com `class_weight='balanced'` quando disponível: Regressão Logística, Decision Tree, KNN e Random Forest. A Regressão Logística apresentou o melhor PR-AUC, indicando uma relação predominantemente linear entre as variáveis e a fraude.

Ao avaliar os modelos em validação com o threshold padrão (0.5), Decision Tree e Random Forest não geraram nenhuma predição positiva. O ROC-AUC de 0.91 do Random Forest mostrou que o modelo separava bem as classes — o problema estava no ponto de corte, não na capacidade discriminativa. O threshold da Regressão Logística foi então recalibrado a partir da curva precision-recall no conjunto de validação, buscando o valor que maximiza o F1.

## Resultado final (conjunto de teste)

| Métrica | Valor |
|---|---|
| PR-AUC | 0.35 |
| Precisão (fraude) | 0.55 – 0.58 |
| Recall (fraude) | 0.24 – 0.49 |
| ROC-AUC | ≈ 0.93 |

Os sinais mais relevantes de fraude identificados pelo modelo foram `merchant_risk_score`, `velocity_score`, `cvv_retry_count`, ausência de autenticação e `ip_country_mismatch`.

O modelo final é salvo junto com o `one_hot` encoder, o `scaler` e o threshold ajustado — sem esses três componentes, o `.pkl` sozinho não é suficiente para prever em dados novos.

## Estrutura

```text
credit_fraud_detection/
├── README.md
├── main.ipynb
└── models/
    └── modelo_fraude_regressao_logistica.pkl
```

## Stack

Python, Pandas, Scikit-Learn, Matplotlib, Seaborn.
