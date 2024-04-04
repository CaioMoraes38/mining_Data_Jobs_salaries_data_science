import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('0-Datasets/DatasetJobsScienceDados.csv')
    print(df.head(15))
    
    
    missing_values = df.isnull().sum()
    # VERIFICAR SE HÁ DADOS FALTANTES NA BASE DE DADOS
    print("\nDados faltantes:\n")
    print(missing_values)

    colunas_features = ['work_year','job_title','job_category','salary_in_usd','company_size',
    'work_setting']

    """
    A BASE DE DADOS NÃO TEM DADOS FALTANTES ENTÃO UM TRATAMENTO NÃO É NECESSARIO,
    MAS COMO SÃO 12 COLUNAS DECIDE TRABALHAR COM METADE DELAS, AS QUE CONSIDEREI MAIS 
    RELEVANTES PARA ANÁLISE
    """

    # CRIA UM NOVO DATAFRAME COM AS COLUNAS DESEJADAS
    df_features = df[colunas_features]
    
    
    print("\nVariações de dados nas Colunas: \n")
    print(df.nunique())

    # SALVAR O ARQUIVO
    df_features.to_csv('0-Datasets/DatasetJobsScienceDadosClear.csv',index=False)    


if __name__ == "__main__":
    main()