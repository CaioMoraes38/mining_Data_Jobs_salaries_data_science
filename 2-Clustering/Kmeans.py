import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Função de KMeans do zero
def KMeans_scratch(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Escolhendo centroids aleatoriamente
    centroids = x[idx, :]  # Passo 1
    
    for _ in range(no_of_iterations):
        # Encontrando a distância entre os centroids e todos os pontos de dados
        distances = cdist(x, centroids, 'euclidean')  # Passo 2
        # Atribuindo o ponto ao centroid mais próximo
        points = np.array([np.argmin(i) for i in distances])  # Passo 3
        
        # Atualizando os centroids
        new_centroids = []
        for idx in range(k):
            if np.any(points == idx):  # Evita divisão por zero se algum cluster estiver vazio
                temp_cent = x[points == idx].mean(axis=0)
            else:
                temp_cent = centroids[idx]  # Mantém o centroide se o cluster estiver vazio
            new_centroids.append(temp_cent)
        
        centroids = np.vstack(new_centroids)  # Centroids atualizados
    
    return points

def plot_samples(projected, labels, title):
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1], label=i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', len(u_labels)))
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.title(title)

def main():
    # Carregar o conjunto de dados personalizado
    data = pd.read_csv('0-Datasets/DatasetJobsScienceDadosClear.csv')  # Ajuste o caminho do arquivo

    # Convertendo colunas categóricas em numéricas se necessário
    data = pd.get_dummies(data, drop_first=True)

    # Selecione as colunas numéricas para o modelo
    numeric_cols = data.columns
    X = data[numeric_cols]

    # Normalizar os dados
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA para redução de dimensionalidade
    pca = PCA(2)
    projected = pca.fit_transform(X_scaled)
    print("Variância explicada por cada componente:", pca.explained_variance_ratio_)
    print("Formato dos dados projetados:", projected.shape)

    # Aplicando a função KMeans do zero
    k = 6  # Defina o número de clusters desejado
    labels = KMeans_scratch(projected, k, 10)
    
    # Visualizar os resultados
    plot_samples(projected, labels, 'Rótulos dos Clusters KMeans do Zero')

    # Aplicando KMeans do sklearn
    kmeans = KMeans(n_clusters=k).fit(projected)
    print("Inércia:", kmeans.inertia_)
    centers = kmeans.cluster_centers_
    score = silhouette_score(projected, kmeans.labels_)
    print(f"Para n_clusters = {k}, a pontuação silhouette é {score:.2f}")

    # Visualizar os resultados do sklearn
    plot_samples(projected, kmeans.labels_, 'Rótulos dos Clusters KMeans do sklearn')

    plt.show()

if __name__ == "__main__":
    main()
