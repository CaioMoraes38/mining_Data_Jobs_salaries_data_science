import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Fazer previsões no conjunto de teste
    y_pred = clf.predict(X_test)
    
    # Calcular e exibir a acurácia no conjunto de teste
    accuracy_test = accuracy_score(y_test, y_pred)
    print(f'Acurácia no conjunto de teste: {accuracy_test:.2f}')
    
    # Avaliação com validação cruzada
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f'Acurácia (Validação Cruzada): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusão:')
    print(cm)

    # Visualizar a matriz de confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
    plt.xlabel('Rótulo Predito')
    plt.ylabel('Rótulo Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()
    
    # Visualizar a árvore de decisão
    plt.figure(figsize=(12,8))
    feature_names = list(X.columns)
    class_names = list(map(str, y.unique()))  
    tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

if __name__ == "__main__":
    main()
