# Importação dos módulos
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Calcula a distância de Minkowski entre dois pontos
def minkowski_distance(a, b, p=1):    
    dim = len(a)    
    distance = 0
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
    return distance**(1/p)

# Previsão usando KNN implementado do zero
def knn_predict(X_train, X_test, y_train, y_test, k, p):    
    y_hat_test = []
    for test_point in X_test:
        distances = []
        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        df_dists = pd.DataFrame(data=distances, columns=['dist'], index=y_train.index)
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        counter = Counter(y_train[df_nn.index])
        prediction = counter.most_common()[0][0]
        y_hat_test.append(prediction)
        
    return y_hat_test

# Plota a matriz de confusão
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

def main():
    # Carrega sua base de dados
    df = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')  # Ajuste o caminho e o formato conforme necessário

    # Verifique as colunas disponíveis
    print("Colunas no DataFrame:", df.columns)

    # Ajuste o nome da coluna de rótulo se necessário
    target_column = 'salary_category'  # Substitua pelo nome correto da coluna de rótulo
    if target_column not in df.columns:
        raise ValueError(f"Coluna de rótulo '{target_column}' não encontrada no DataFrame.")

    # Codifica variáveis categóricas usando one-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    print("Colunas após codificação:", df_encoded.columns)

    # Separa os dados em X e y novamente
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]
    print("Total samples: {}".format(X.shape[0]))

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Normaliza os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
        
    # Teste usando KNN implementado do zero
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy K-NN from scratch: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scratch: {:.2f}".format(f1))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, np.unique(y), False, "Confusion Matrix - K-NN")
    plot_confusion_matrix(cm, np.unique(y), True, "Confusion Matrix - K-NN normalized")  

    # Teste usando KNN do sklearn
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy K-NN from sklearn: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from sklearn: {:.2f}".format(f1))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, np.unique(y), False, "Confusion Matrix - K-NN sklearn")
    plot_confusion_matrix(cm, np.unique(y), True, "Confusion Matrix - K-NN sklearn normalized")
    
    plt.show()

if __name__ == "__main__":
    main()
