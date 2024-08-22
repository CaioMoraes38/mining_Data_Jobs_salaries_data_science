import pandas as pd
from math import sqrt

# Função para calcular a distância euclidiana entre dois vetores
def distancia_euclideana(vet1, vet2):
    distancia = 0
    for i in range(len(vet1)-1):  # Excluímos o último valor que é o rótulo
        distancia += (vet1[i] - vet2[i])**2
    return sqrt(distancia)

# Função para retornar os k vizinhos mais próximos
def retorna_vizinhos(base_treinamento, amostra_teste, num_vizinhos):
    distancias = list()
    for linha_tre in base_treinamento:
        dist = distancia_euclideana(amostra_teste, linha_tre)
        distancias.append((linha_tre, dist))
    distancias.sort(key=lambda tup: tup[1])  # Ordenando distâncias de forma crescente
    vizinhos = list()
    for i in range(num_vizinhos):
        vizinhos.append(distancias[i][0])
    return vizinhos

# Função de predição/classificação
def classifica(base_treinamento, amostra_teste, num_vizinhos):
    vizinhos = retorna_vizinhos(base_treinamento, amostra_teste, num_vizinhos)
    rotulos = [v[-1] for v in vizinhos]  # Último valor em cada linha é o rótulo
    predicao = max(set(rotulos), key=rotulos.count)
    return predicao

def main():
    # Carregando o conjunto de dados
    data = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')

    # Convertendo colunas categóricas em numéricas usando o método de codificação de rótulos
    data['job_title'] = data['job_title'].astype('category').cat.codes
    data['job_category'] = data['job_category'].astype('category').cat.codes
    data['work_setting'] = data['work_setting'].astype('category').cat.codes
    data['company_size'] = data['company_size'].astype('category').cat.codes

    # Definindo as colunas numéricas
    numeric_cols = ['work_year', 'salary_in_usd', 'job_title', 'job_category', 'company_size', 'work_setting', 'salary_category']
    data_numeric = data[numeric_cols]

    # Adicionando uma coluna de rótulos (neste caso, usaremos a 'salary_category' como rótulo)
    labels = data['salary_category']
    dataset = data_numeric.values.tolist()

    # Selecionando uma amostra para testar
    amostra = dataset[0]  # A primeira amostra do dataset

    # Executando a classificação usando KNN
    predicao = classifica(dataset, amostra, 3)
    print('Resultado da classificação')
    print('Esperado %d\nPredição %d' % (amostra[-1], predicao))

if __name__ == "__main__":
    main()
