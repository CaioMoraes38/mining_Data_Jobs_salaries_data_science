import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def main():
    
    df_clear = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')
    print(df_clear.head(15))

   
    numeric_columns = df_clear.select_dtypes(include=[np.number]).columns
    
    scaler = MinMaxScaler()
    df_clear[numeric_columns] = scaler.fit_transform(df_clear[numeric_columns]) 

    df_clear.to_csv('0-Datasets/DatasetJobsScienceDadosClear_Normalized.csv', index=False)

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
