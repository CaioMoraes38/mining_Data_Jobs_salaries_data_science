import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Função KMeans implementada do zero
def KMeans_scratch(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    centroids = x[idx, :]  # Passo 1: Escolhendo centroides aleatoriamente

    for _ in range(no_of_iterations):
        distances = cdist(x, centroids, 'euclidean')  # Passo 2: Calculando distâncias
        points = np.array([np.argmin(i) for i in distances])  # Passo 3: Atribuindo pontos aos clusters mais próximos

        centroids = []
        for idx in range(k):
            # Passo 4: Atualizando os centroides
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Atualizando os centroides para a próxima iteração

    return points

# Função para plotar amostras
def plot_samples(projected, labels, title):
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1], label=i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title(title)

def main():
    # Carregando o conjunto de dados
    data = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')

    # Convertendo colunas categóricas em numéricas usando o método de codificação de rótulos
    data['job_title'] = data['job_title'].astype('category').cat.codes
    data['job_category'] = data['job_category'].astype('category').cat.codes
    data['work_setting'] = data['work_setting'].astype('category').cat.codes
    data['company_size'] = data['company_size'].astype('category').cat.codes

    # Selecione apenas as colunas numéricas para o modelo
    numeric_cols = ['work_year', 'salary_in_usd', 'job_title', 'job_category', 'company_size', 'work_setting', 'salary_category']
    data_numeric = data[numeric_cols]

 
    pca = PCA(2)
    projected = pca.fit_transform(data_numeric)
    print("Variância explicada por cada componente:", pca.explained_variance_ratio_)
    print("Formato original dos dados:", data_numeric.shape)
    print("Formato dos dados projetados:", projected.shape)
    
    # Aplicando a função KMeans implementada do zero
    labels_scratch = KMeans_scratch(projected, 4, 4)
    
    # Visualizando os resultados do KMeans implementado do zero
    plot_samples(projected, labels_scratch, 'Clusters Labels KMeans from scratch')

    # Aplicando a função KMeans do sklearn
    kmeans = KMeans(n_clusters=4).fit(projected)
    score = silhouette_score(projected, kmeans.labels_)
    print("Silhouette score para n_clusters=6:", score)

    # Visualizando os resultados do KMeans do sklearn
    plot_samples(projected, kmeans.labels_, 'Clusters')

    plt.show()

if __name__ == "__main__":
    main()
