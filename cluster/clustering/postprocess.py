import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

class ClusterPostProcessor:
    def __init__(self, config):
        self.config = config
        
    def merge_similar_clusters(self, embeddings, labels, threshold=0.85):
        """合并相似簇"""
        valid_labels = [l for l in np.unique(labels) if l != -1]
        
        # 检查是否有有效的簇
        if len(valid_labels) == 0:
            return labels
        
        centroids = [embeddings[labels == l].mean(axis=0) for l in valid_labels]
        
        # 构建相似度矩阵
        sim_matrix = cosine_similarity(centroids)
        
        # 合并逻辑
        label_mapping = {l: l for l in valid_labels}
        label_mapping[-1] = -1  # 为噪声标签添加映射
        for i in range(len(valid_labels)):
            for j in range(i+1, len(valid_labels)):
                if sim_matrix[i][j] > threshold:
                    label_mapping[valid_labels[j]] = valid_labels[i]
        
        # 指定默认值，防止返回 None
        return np.vectorize(lambda x: label_mapping.get(x, x))(labels)
    
    def handle_noise_points(self, embeddings, labels):
        """处理噪声点"""
        noise_mask = (labels == -1)
        if not np.any(noise_mask):
            return labels
        
        # 找到最近的簇心
        valid_embeddings = embeddings[~noise_mask]
        valid_labels = labels[~noise_mask]
        
        # 检查是否有有效的样本
        if valid_embeddings.shape[0] == 0:
            return labels
        
        knn = NearestNeighbors(n_neighbors=5).fit(valid_embeddings)
        _, indices = knn.kneighbors(embeddings[noise_mask])
        
        # 分配最常见的标签
        new_labels = labels.copy()
        for idx, neighbors in zip(np.where(noise_mask)[0], indices):
            new_labels[idx] = np.bincount(valid_labels[neighbors]).argmax()
            
        return new_labels