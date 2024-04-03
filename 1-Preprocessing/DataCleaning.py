import pandas as pd
import numpy as np

def main():

    df = pd.read_csv('0-Datasets/DatasetJobsScienceDados.csv')
    print(df.head(30))

    missing_values = df.isnull().sum()

    # VERIFICAR SE HÁ DADOS FALTANTES NA BASE DE DADOS
    print("\nDados faltantes:\n")
    print(missing_values)

    colunas_features = ['work_year','job_title','job_category','salary_in_usd',
    'work_setting','experience_level']

    # CRIA UM NOVO DATAFRAME COM AS COLUNAS DESEJADAS
    df_features = df[colunas_features]

    # SALVAR O ARQUIVO
    df_features.to_csv('0-Datasets/DatasetJobsScienceDadosClear.csv', index=False) 

    df_features.describe()

if __name__ == "__main__":
    main()