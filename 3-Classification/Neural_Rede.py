import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    # Carrega o dataset
    df = pd.read_csv(file_path)
    
    # Verifica se o DataFrame foi carregado corretamente
    if df.empty:
        raise ValueError("O DataFrame está vazio. Verifique o caminho do arquivo e o conteúdo do arquivo CSV.")
    
    # Assumindo que 'salary_category' é o alvo
    target_column = 'salary_category'
    if target_column not in df.columns:
        raise ValueError(f"Coluna de rótulo '{target_column}' não encontrada no DataFrame.")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Verifica se há valores ausentes e os trata
    if X.isnull().values.any():
        X = X.fillna(X.mean())  # Preenche valores ausentes com a média da coluna
    
    # Codifica variáveis categóricas, se houver
    X = pd.get_dummies(X, drop_first=True)
    
    # Codifica a variável alvo
    y = y.astype('category').cat.codes

    return X, y

def build_and_train_nn(X_train, y_train):
    # Define o modelo da rede neural
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Camada de entrada e primeira camada oculta
    model.add(Dense(32, activation='relu'))  # Segunda camada oculta
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Camada de saída (para classificação)

    # Compila o modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Treina o modelo
    history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=1)
    
    return model, history

def main():
    # Carrega e pré-processa os dados
    X, y = load_and_preprocess_data('0-Datasets/Normatização_Min-Max_DatasetJobsScienceDadosNumericos.csv')

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Normaliza os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cria e treina o modelo da rede neural
    model, history = build_and_train_nn(X_train, y_train)

    # Avalia o modelo
    y_pred = np.argmax(model.predict(X_test), axis=-1)  # Converte as previsões em classes

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusão:')
    print(cm)

    # Relatório de Classificação
    print('Relatório de Classificação:')
    print(classification_report(y_test, y_pred))

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print('Acurácia:', accuracy)

    # Visualiza o histórico de treinamento
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(['Treinamento', 'Validação'])
    plt.show()

    # Visualiza a Matriz de Confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Rótulo Predito')
    plt.ylabel('Rótulo Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()

if __name__ == "__main__":
    main()