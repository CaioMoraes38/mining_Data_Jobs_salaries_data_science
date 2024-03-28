import pandas as pd     # Biblioteca para manipulação de dados em formato tabular
import numpy as np      # Biblioteca para cálculos numéricos
import matplotlib.pyplot as plt    # Biblioteca para criação de gráficos
import seaborn as sns   # Biblioteca para visualização de dados
import matplotlib.font_manager as fm

def create_bins(df, column_name, bins, labels):
    df[column_name + '_bins'] = pd.cut(df[column_name], bins=bins, labels=labels)
    return df

def PlotBarChart(df, column_name_num):
    """
    Plota um gráfico de contagem das faixas de idade no dataframe df.
    """  
    # Verificar valores ausentes no dataframe
    if df.isnull().values.any():
        print("Atenção: existem valores ausentes no dataframe.")
        
    
    # Definir propriedades de fonte
    title_font = fm.FontProperties(weight='bold', size=20)

    # Set the plot title, x-axis limit and show the plot
    plt.title(f'No Stroke vs Stroke by {column_name_num}',fontproperties=title_font)
    plt.xlabel(column_name_num)
    plt.legend()
    plt.grid(True)
   
     
def PlotPieChart(df, column_name_cat):
    # Crie um gráfico em formato de pizza com a contagem de cada fruta
    fatias, texto, autotexto = plt.pie(df[column_name_cat].value_counts(), autopct='%.2f%%')

    # Definir propriedades de fonte
    title_font = fm.FontProperties(weight='bold', size=20)

    # Adicione rótulos nas fatias
    for fatia in fatias:
        fatia.set_label('{}'.format(fatia.get_label()))

    # Adicione legendas
    plt.legend(fatias, df[column_name_cat].unique(), loc='best')

    # Adicione título e formatação final
    plt.title(f'Distribution of {column_name_cat}',fontproperties=title_font)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"previsao_AVC/9-Graphics/{column_name_cat}_pizza.png")
    
def main():
    names = ["work_year","job_title","job_category","salary_currency","salary","salary_in_usd",
             "employee_residence","experience_level","employment_type","work_setting","company_location"
             ,"company_size"] 
    features = ['work_year','job_title','salary_currency','salary','company_location','company_size']
    output_file ='1-DataBase/DatasetJobsScienceDadosClear.csv'
    input_file ='1-DataBase/DatasetJobsScienceDados.csv'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes

   
    print(df.describe().T)
    

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
    cat = []
    num = []
    for i in df.columns:
        if df[i].dtypes == object:
            cat.append(i)
        else :
            num.append(i)
    for column_name_cat in cat:
        PlotPieChart(df, column_name_cat)  
    bins = []
    labels = []
    for column_name_num in num:   
        # Criar faixas de 
        if column_name_num == "age":
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"]
        # Criar faixas de indice de massa corporal
        elif column_name_num == "bmi":
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
        # Criar faixas de avg_glucose_level
        elif column_name_num == "avg_glucose_level":
            bins = [0, 50, 100, 150, 200, 250, 300]
            labels = ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300"]

        df = create_bins(df, column_name_num, bins, labels)     
        PlotBarChart(df, column_name_num)
    
if __name__ == "__main__":
    main()