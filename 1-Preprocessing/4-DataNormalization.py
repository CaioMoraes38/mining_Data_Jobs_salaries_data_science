import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def main():
    # Carregar o DataFrame
    df_clear = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')
    print(df_clear.head(15))

    # Selecionar apenas as colunas numéricas
    numeric_columns = df_clear.select_dtypes(include=[np.number]).columns

    # Normalizar apenas as colunas numéricas usando Min-Max
    scaler = MinMaxScaler()
    df_clear[numeric_columns] = scaler.fit_transform(df_clear[numeric_columns])

    # Salvar o DataFrame normalizado em um arquivo CSV
    df_clear.to_csv('0-Datasets/DatasetJobsScienceDadosClear_Normalized.csv', index=False)

    # Plotar histogramas das colunas normalizadas
    plt.figure(figsize=(8, 6))
    plt.hist(df_clear['work_year'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Histograma do Ano de Trabalho Normalizado (Min-Max)')
    plt.xlabel('Ano de Trabalho Normalizado')
    plt.ylabel('Frequência')
    plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.hist(df_clear['salary_in_usd'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Salario em USD - (Min-Max)')
    plt.xlabel('Salarios')
    plt.ylabel('Frequência')
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
