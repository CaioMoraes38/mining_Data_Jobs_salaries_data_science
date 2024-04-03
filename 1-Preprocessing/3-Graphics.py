import pandas as pd
import matplotlib.pyplot as plt

def main():
    df_clear = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')
    print(df_clear.head(15))

   
    Ano = df_clear['work_year'].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    Ano.plot(kind='bar', color='lightgreen')
    plt.title('Distribuição do Ano de Trabalho')
    plt.xlabel('Ano de Trabalho')
    plt.ylabel('')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.hist(df_clear['salary_in_usd'], bins=20, color='skyblue')
    plt.title('Histograma da Coluna Salary (USD)')
    plt.xlabel('Salário (USD)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
