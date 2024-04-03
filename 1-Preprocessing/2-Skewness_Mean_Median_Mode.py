import pandas as pd
import numpy as np
import statistics as st


def main():
    df_clear = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')
    print(df_clear.head(15))

    cat = df_clear.select_dtypes(include='object')
    cat.columns
    num = df_clear.select_dtypes(include=np.number)
    num.columns

    print("\nMedia Salarial em Dolares Americanos ao Ano:\n")

    Media = df_clear['salary_in_usd'].mean()
    print('\nA média da coluna "salary_in_usd" é:', Media)
    Assimetria = df_clear["salary_in_usd"].skew()
    print('\nAssimetria da coluna "salary_in_usd" é:',Assimetria)
    Moda = df_clear['salary_in_usd'].mode()
    print('\nModa da coluna "salary_in_usd" é:',Moda)

    for i in cat.columns:
        print("Valor mais frequente na coluna " + i.upper() + " é: " + str(cat[i].mode()[0]))


if __name__ == "__main__":
    main()
