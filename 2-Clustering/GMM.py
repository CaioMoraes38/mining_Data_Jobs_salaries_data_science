import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def plot_samples(projected, labels, title):    
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i , 0] , projected[labels == i , 1] , label = i,
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

    # Aplicando PCA para reduzir a dimensionalidade para visualização
    pca = PCA(2)
    projected = pca.fit_transform(data_numeric)
    print("Variância explicada por cada componente:", pca.explained_variance_ratio_)
    print("Formato original dos dados:", data_numeric.shape)
    print("Formato dos dados projetados:", projected.shape)

    # Testando diferentes números de clusters usando GMM
    n_clusters = range(1, 11)
    bics = []
    aics = []
    
    for n in n_clusters:
        gm = GaussianMixture(n_components=n).fit(projected)
        bics.append(gm.bic(projected))
        aics.append(gm.aic(projected))

    # Plotando BIC e AIC
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters, bics, label='BIC', marker='o')
    plt.plot(n_clusters, aics, label='AIC', marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Valor do Critério')
    plt.title('BIC e AIC para Diferentes Números de Clusters')
    plt.legend()
    plt.show()

    
    optimal_clusters = n_clusters[np.argmin(bics)]
    print(f'Número ideal de clusters baseado no BIC: {optimal_clusters}')

    # Aplicando GMM com o número ideal de clusters
    gm = GaussianMixture(n_components=optimal_clusters).fit(projected)
    labels = gm.predict(projected)

    # Visualizando os resultados
    plot_samples(projected, labels, f'Clusters Labels GMM (n_clusters={optimal_clusters})')

    plt.show()

if __name__ == "__main__":
    main()
