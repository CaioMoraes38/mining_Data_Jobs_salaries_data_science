# Importação dos módulos
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def load_dataset(file_path):        
    df = pd.read_csv(file_path)

    # Assumindo que a coluna de rótulo é 'target'; ajuste se necessário
    target_column = 'salary_category'
    if target_column not in df.columns:
        raise ValueError(f"Coluna de rótulo '{target_column}' não encontrada no DataFrame.")
    
    # Codifica variáveis categóricas usando one-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    print("Colunas após codificação:", df_encoded.columns)

    return df_encoded

def main():
    # Carrega o dataset
    df = load_dataset('0-Datasets/DatasetJobsScienceDadosClear.csv')  # Ajuste o caminho do arquivo

    # Ajuste o nome da coluna de rótulo se necessário
    target_column = 'salary_category'  # Substitua pelo nome correto da coluna de rótulo

    # Separa os dados em X e y
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("Total samples: {}".format(X.shape[0]))

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Normaliza os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
        
    # TESTE USANDO SVM do sklearn    
    svm = SVC(kernel='poly')  # Pode ser 'poly', 'rbf', 'linear'
    # Treinamento usando o conjunto de dados de treino
    svm.fit(X_train, y_train)
    # Predição usando o conjunto de dados de teste
    y_hat_test = svm.predict(X_test)

    # Obtém a acurácia do teste
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy SVM from sklearn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sklearn: {:.2f}".format(f1))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(cm, np.unique(y), False, "Confusion Matrix - SVM sklearn")
    plot_confusion_matrix(cm, np.unique(y), True, "Confusion Matrix - SVM sklearn normalized")
    
    plt.show()

if __name__ == "__main__":
    main()
