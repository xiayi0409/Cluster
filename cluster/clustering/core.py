import numpy as np
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.cluster import OPTICS, SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from config.settings import Settings

class ClusterEngine:
    def __init__(self,config):
        self.settings = config
        
    def two_stage_clustering(self, embeddings):
        """两阶段聚类算法"""
        # 第一阶段：粗粒度聚类
        kmeans = KMeans(**self.config.CLUSTER_METHODS['two_stage']['stage1'])
        stage1_labels = kmeans.fit_predict(embeddings)
        
        # 第二阶段：细粒度聚类
        final_labels = np.zeros(len(embeddings), dtype=int)
        current_max_label = 0
        
        for cluster_id in np.unique(stage1_labels):
            mask = (stage1_labels == cluster_id)
            sub_data = embeddings[mask]
            
            if len(sub_data) < 10:  # 小簇不细分
                final_labels[mask] = cluster_id
                continue
                
            dbscan = DBSCAN(**self.config.CLUSTER_METHODS['two_stage']['stage2'])
            sub_labels = dbscan.fit_predict(sub_data)
            
            # 重新编号避免标签冲突
            sub_labels[sub_labels != -1] += current_max_label
            final_labels[mask] = sub_labels
            current_max_label = final_labels.max() + 1
            
        return final_labels
    
    def hdbscan_clustering(self, embeddings):
        """密度聚类算法"""
        clusterer = hdbscan(**self.config.CLUSTER_METHODS['hdbscan'])
        return clusterer.fit_predict(embeddings)
    
    # 新增OPTICS和谱聚类方法
    def optic_clustering(self, embeddings):
        clusterer = OPTICS(**self.config.CLUSTER_METHODS['optics'])
        return clusterer.fit_predict(embeddings)
    
    def spectral_clustering(self, embeddings):
        clusterer = SpectralClustering(**self.config.CLUSTER_METHODS['spectral'])
        return clusterer.fit_predict(embeddings)

    def cluster(self, method, embeddings):
        config = self.settings.CLUSTER_METHODS.get(method)
        if config is None:
            raise ValueError(f"Unsupported clustering method: {method}")
        if method == 'kmeans':
            clusterer = KMeans(**config)
        elif method == 'dbscan':
            clusterer = DBSCAN(**config)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(**config)
        elif method == 'gmm':
            clusterer = GaussianMixture(**config)
        elif method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(**config)
        elif method == 'spectral':
            clusterer = SpectralClustering(**config)
        elif method == 'optics':
            clusterer = OPTICS(**config)
        elif method == 'two_stage':
            labels = self.two_stage_clustering(embeddings)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        labels = clusterer.fit_predict(embeddings)
        return labels
