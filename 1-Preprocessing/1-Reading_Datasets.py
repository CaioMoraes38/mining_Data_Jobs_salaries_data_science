import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('0-Datasets/DatasetJobsScienceDados.csv')
    print(df.head(15))
    
    missing_values = df.isnull().sum()
    print("\nDados faltantes:\n")
    print(missing_values)

    colunas_features = ['work_year', 'job_title', 'job_category', 'salary_in_usd', 'company_size', 'work_setting']

    df_features = df[colunas_features]
    
    print("\nVariações de dados nas Colunas: \n")
    print(df_features.nunique())

    # Categorizar a coluna salary_in_usd em 4 categorias
    df_features['salary_category'] = pd.cut(df_features['salary_in_usd'], bins=4, labels=[1, 2, 3, 4])
    
    print("\nDados categorizados:\n")
    print(df_features.head(15))

    # SALVAR O ARQUIVO
    df_features.to_csv('0-Datasets/DatasetJobsScienceDadosClear.csv', index=False)    

if __name__ == "__main__":
    main()
