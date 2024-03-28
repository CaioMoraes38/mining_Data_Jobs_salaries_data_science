import pandas as pd     # Biblioteca para manipulação de dados em formato tabular

def main():
    # Definição dos nomes das colunas e características do arquivo
    names = ["work_year","job_title","job_category","salary_currency","salary","salary_in_usd",
             "employee_residence","experience_level","employment_type","work_setting","company_location",
             "company_size"] 
    features = ['work_year','job_title','salary_currency','salary','company_location','company_size']
    original_output_file ='1-DataBase/DatasetJobsScienceDadosClear.csv'  
    new_output_file ='1-DataBase/Dataset_Dados_Numericos.csv'    
    input_file ='1-DataBase/DatasetJobsScienceDados.csv'

    try:
        # Leitura do arquivo CSV
        df = pd.read_csv(input_file,         # Nome do arquivo com dados
                         names = names,      # Nome das colunas 
                         usecols = features, # Define as colunas que serão utilizadas
                         na_values='?')      # Define que ? será considerado valores ausentes
        
        # Imprime a quantidade de valores faltantes por coluna
        print("\nVALORES FALTANTES\n")
        print(df.isnull().sum())
        print("\n")   

      
        print(df.describe().T)

        print("\nDADOS CATEGÓRICOS PARA VALORES NUMÉRICOS\n")   

        
        for coluna in features:
            contagem = df[coluna].value_counts()    
            df_contagem = contagem.to_frame().reset_index()   
            df_contagem.columns = [coluna, "Incidencia"]    
            print(df_contagem)    

      
        df_numerical = df.copy()
        df_numerical['work_year'] = df_numerical['work_year'].replace({'2020': 0, '2021': 1, '2022': 2,'2023':3})

       
        ShowInformationDataFrame(df_numerical, "\nDataframe Dados Numéricos")           
        df.to_csv(original_output_file, header=False, index=False)  
        df_numerical.to_csv(new_output_file, header=False, index=False)  
        
        print("Arquivos salvos com sucesso:", original_output_file, new_output_file)
        
    except Exception as e:
        print("Ocorreu um erro ao salvar os arquivos:", e)

def ShowInformationDataFrame(df, description):
    print(description)
    print(df.info())
    print(df.describe())
    print("\n")

if __name__ == "__main__":
    main()
