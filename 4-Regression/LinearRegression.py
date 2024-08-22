import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    # Carrega o dataset
    df = pd.read_csv(file_path)

    # Assumindo que a última coluna é o alvo
    target_column = df.columns[-1]  # Ajuste se necessário
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Codifica variáveis categóricas, se houver
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def main():
    # Carrega e pré-processa os dados
    X, y = load_and_preprocess_data('0-Datasets/Normatização_Min-Max_DatasetJobsScienceDadosNumericos.csv')  

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Normaliza os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regr = LinearRegression()
    regr.fit(X_train, y_train)

    cv_scores = cross_val_score(regr, X_train, y_train, cv=5, scoring='r2')
    print(f'R2 Score (Cross-Validation): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')

    r2_train = regr.score(X_train, y_train)
    r2_test = regr.score(X_test, y_test)
    print('R2 no set de treino: %.2f' % r2_train)
    print('R2 no set de teste: %.2f' % r2_test)

    y_pred = regr.predict(X_test)
    abs_error = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    print('Erro absoluto no set de teste: %.2f' % abs_error)
    print('Erro quadrático médio no set de teste: %.2f' % mse)
    print('Raiz do erro quadrático médio no set de teste: %.2f' % rmse)

    # Visualizar os resultados
    # Gráfico de Dispersão das Previsões vs. Valores Reais
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title('Previsões vs. Valores Reais')

    # Gráfico dos Erros de Previsão
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduos')
    plt.title('Gráfico de Resíduos')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
