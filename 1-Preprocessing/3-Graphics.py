import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def VisualizarGraficos(df, target, ano):
    # Configuração do gráfico de barras para o ano de trabalho
    sns.countplot(x='work_year',data=df)
    plt.title('Ano Trabalho')
    plt.xlabel('Anos')
    plt.show()
    
    # Gráfico de pizza para categorias de trabalho
    top_categories = df[target].value_counts().nlargest(4)
    other_count = df[target].value_counts().drop(top_categories.index).sum()
    combined_counts = pd.concat([top_categories, pd.Series({'Outros': other_count})])
    plt.figure(figsize=(8, 8))
    combined_counts.plot(kind='pie', autopct='%0.2f%%')
    plt.title('Categorias de Trabalho')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    # Histograma da coluna Salary (USD)
    plt.figure(figsize=(8, 6))
    plt.hist(df['salary_in_usd'], bins=20, color='skyblue')
    plt.title('Histograma da Coluna Salary (USD)')
    plt.xlabel('Salário (USD)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()

def main():
    df_clear = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')
    target = 'job_category'
    print(df_clear.head(15))
    
    # Convertendo os anos para inteiros antes de contar a distribuição dos anos
    ano = df_clear['work_year'].astype(int).value_counts().sort_index()
   
    VisualizarGraficos(df_clear, target, ano)

if __name__ == "__main__":
    main()
