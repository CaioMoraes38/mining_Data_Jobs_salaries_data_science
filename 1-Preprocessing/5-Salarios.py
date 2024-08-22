import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Carrega o dataset
    df = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')
    
    # Verifica dados faltantes
    missing_values = df.isnull().sum()
    print("\nDados faltantes:\n")
    print(missing_values)

    # Seleciona colunas relevantes
    colunas_features = ['work_year', 'job_title', 'job_category', 'salary_in_usd', 'company_size', 'work_setting', 'salary_category']

    # Cria um novo DataFrame com as colunas desejadas
    df_features = df[colunas_features]

    # Converte colunas categóricas em numéricas usando codificação de rótulos
    df_features['job_title'] = df_features['job_title'].astype('category').cat.codes
    df_features['job_category'] = df_features['job_category'].astype('category').cat.codes
    df_features['company_size'] = df_features['company_size'].astype('category').cat.codes
    df_features['work_setting'] = df_features['work_setting'].astype('category').cat.codes

    print("\nVariações de dados nas Colunas: \n")
    print(df_features.nunique())

    # Salva o arquivo
    df_features.to_csv('0-Datasets/DatasetJobsScienceDadosNumericos.csv', header=True, index=False)    
   
    # Gráfico de pizza para as categorias de salário
    plt.figure(figsize=(8, 8))
    labels = ['1 - 112.500', '2 - 112.501 a 225.000', '3 - 225.001 a 337.500', '4 - 337.501 a 450.000']  # Rótulos para a legenda
    df['salary_category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set3', wedgeprops={'edgecolor': 'black'})
    plt.title('Salário anual USD')
    plt.ylabel('')  

    # Adiciona a legenda com posição personalizada
    plt.legend(labels, title="Categorias", loc="upper left", bbox_to_anchor=(1, 0.5))

    plt.show()

if __name__ == "__main__":
    main()
