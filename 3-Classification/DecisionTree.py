import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

def main():
    # Carregar o conjunto de dados
    data = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')
    
    # Exibir as primeiras linhas e colunas do DataFrame para entender os dados
    print(data.head())
    print(data.columns)

    # Convertendo colunas categóricas em numéricas
    data['job_title'] = data['job_title'].astype('category').cat.codes
    data['job_category'] = data['job_category'].astype('category').cat.codes
    data['work_setting'] = data['work_setting'].astype('category').cat.codes
    data['company_size'] = data['company_size'].astype('category').cat.codes
    
    
    if 'salary_category' in data.columns:
        X = data.drop('salary_category', axis=1)
        y = data['salary_category']
    else:
        print("A coluna 'salary_category' não está disponível. Ajuste a escolha do rótulo conforme necessário.")
        return
    
    # Dividir os dados - 70% treino, 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Criar e treinar o classificador de árvore de decisão
    clf = DecisionTreeClassifier(max_leaf_nodes=200)
    clf.fit(X_train, y_train)
    
    # Visualizar a árvore de decisão
    plt.figure(figsize=(12,8))
    feature_names = list(X.columns)
    class_names = list(map(str, y.unique()))  
    tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

if __name__ == "__main__":
    main()
