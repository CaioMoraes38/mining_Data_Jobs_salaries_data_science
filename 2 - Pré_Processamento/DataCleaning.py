import pandas as pd     # Biblioteca para manipulação de dados em formato tabular
import numpy as np      # Biblioteca para cálculos numéricos
import matplotlib.pyplot as plt    # Biblioteca para criação de gráficos
import seaborn as sns   # Biblioteca para visualização de dados
import matplotlib.font_manager as fm

    
def main():
    names = ["work_year","job_title","job_category","salary_currency","salary","salary_in_usd",
             "employee_residence","experience_level","employment_type","work_setting","company_location"
             ,"company_size"] 
    features = ['work_year','job_title','salary_currency','salary','company_location','company_size','work_setting']
    output_file ='1-DataBase/DatasetJobsScienceDadosClear.csv'
    input_file ='1-DataBase/DatasetJobsScienceDados.csv'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes
                     
   # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")   
    print(df.describe().T)

    columns_missing_value = df.columns[df.isnull().any()]
    print(columns_missing_value)
    method = 'mode' # number or median or mean or mode
    df.to_csv(output_file, header=False, index=False)
    

    print(df.describe(include=object).T)
   
    print(f"Data has {df.shape[0]} instances and {df.shape[1] - 1} attributes.")
    
    '''
    Mostrar todos os valores e frequências de uma coluna categórica  
    Para fazer isso, você pode seguir o exemplo abaixo:
    '''
    # 1. Escolha a coluna que você deseja examinar
    for coluna in features:

        # 2.método value_counts() para contar o número de ocorrências 
        contagem = df[coluna].value_counts()

      
        df_contagem = contagem.to_frame().reset_index()

        # 4. Renomeie as colunas para refletir o que elas representam
        df_contagem.columns = [coluna, "Incidencia"]

       
        print(df_contagem)
if __name__ == "__main__":
    main()