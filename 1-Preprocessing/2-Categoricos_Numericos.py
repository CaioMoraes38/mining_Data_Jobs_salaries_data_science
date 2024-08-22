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
    colunas_features = ['work_year','job_title','job_category','salary_in_usd','company_size','work_setting','salary_category']

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
    df_features.to_csv('0-Datasets/DatasetJobsScienceDadosNumericos.csv',header=True, index=False)    
   
    correlacao = df_features.corr()
    
    # Cria o mapa de calor
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Matriz da Correlação')
    plt.yticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    main()
