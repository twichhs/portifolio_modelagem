# Portfólio de Modelagem e Ciência de Dados

meu portfólio central de Data Science e Machine Learning. Este repositório é dedicado a projetos aplicados à resolução de problemas de negócios, com foco especial em **modelagem de risco de crédito, precificação (pricing) e análises geoespaciais**.


---

## 🛠️ Ferramentas e Stack Tecnológico

Minhas soluções são desenvolvidas com foco em performance, automação de pipelines e rigor analítico, utilizando o seguinte ecossistema:

*   **Linguagens:** Python, SQL
*   **Manipulação e Análise de Dados:** Pandas,Polars e NumPy
*   **Análise Geoespacial:** GeoPandas (análises com lat/long e polígonos de dados públicos do IBGE)
*   **Machine Learning e Deep Learning:** Scikit-Learn, TensorFlow, Keras
---

## 📂 Projetos em Destaque

Abaixo estão os principais projetos focados em modelagem. Clique no título de cada um para acessar o repositório detalhado, o código-fonte e as conclusões das análises.

### 1. [Modelagem de Risco de Crédito (Credit Scoring)](#)
* **Objetivo:** Desenvolver um modelo de classificação para prever a probabilidade de inadimplência (PD - *Probability of Default*) de novos solicitantes de crédito.
* **Técnicas Utilizadas:** Teste de hipóteses para seleção de variáveis, tratamento de dados desbalanceados e modelo de Decision Tree.
* **Impacto de Negócio:** Redução simulada da taxa de aprovação de maus pagadores em X%, maximizando o lucro esperado da carteira.
* **Stack:** Python, Pandas, Scikit-Learn, Matplotlib e Seaborn.

---

## 🏗️ Padrão de Arquitetura dos Projetos

Para garantir a reprodutibilidade e organização, todos os projetos individuais listados acima seguem a seguinte estrutura de diretórios:

```text
├── nome_projeto/
│   ├── main.ipynb # analise exploratória a modelagem
│   ├── data/ # datasets
|   ├── models/ # saida dos modelos plk
