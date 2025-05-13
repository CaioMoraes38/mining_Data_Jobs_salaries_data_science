# 📦 Projeto de Machine Learning

Este repositório organiza um pipeline completo de aprendizado de máquina, com foco em análise exploratória, preparação de dados e aplicação de algoritmos de clustering, classificação e regressão.

## 📁 Estrutura das Pastas

- **0-Datasets/**  
  Contém os conjuntos de dados utilizados em todo o projeto. São os dados brutos, sem tratamento.

- **1-Preprocessing/**  
  Scripts de pré-processamento dos dados:
  - Limpeza de dados (remoção de nulos, valores inconsistentes)
  - Normalização/Padronização
  - Codificação de variáveis categóricas
  - Divisão treino/teste

- **2-Clustering/**  
  Técnicas de aprendizado não supervisionado, como:
  - K-Means
  - DBSCAN
  - Agrupamento Hierárquico

- **3-Classification/**  
  Modelos de classificação supervisionada, como:
  - Regressão Logística
  - Decision Tree
  - Random Forest
  - SVM
  - Redes Neurais

- **4-Regression/**  
  Modelos supervisionados para prever valores contínuos:
  - Regressão Linear
  - Ridge/Lasso
  - Regressão com árvores
  - Regressão com redes neurais

## 🧰 Requisitos

Certifique-se de ter o Python 3.8+ instalado. As bibliotecas principais incluem:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
