# %% utils/visualization.py
# utils/visualization.py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP  # 如果前面注释了umap需要取消
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt

def plot_clusters(embeddings, labels, method='umap', save_path=None):
    if method == 'umap':
        reducer = UMAP(n_neighbors=5, n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
        
    coords = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                        c=labels, cmap='tab20', s=10)
    plt.title(f"Clustering Visualization ({method.upper()})")
    plt.colorbar(scatter)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"聚类可视化已保存至: {save_path}")
    plt.close()

def plot_elbow_curve(embeddings, max_k=10):
    """绘制K-means肘部曲线"""
    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k+1), sse, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.show()
