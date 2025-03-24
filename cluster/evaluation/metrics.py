import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import math

class ClusterEvaluator:
    def __init__(self, config):
        self.config = config
    
    # 新增多维度指标计算
    def calculate_all_metrics(self, embeddings, labels, elapsed_time):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters < 2:
            davies_bouldin = np.nan
        else:
            davies_bouldin = davies_bouldin_score(embeddings, labels)

        return {
            'time': elapsed_time,
            'n_clusters': n_clusters,
            'noise_ratio': np.sum(labels == -1) / len(labels) if -1 in labels else 0,
            'silhouette': silhouette_score(embeddings, labels) if n_clusters > 1 else np.nan,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz_score(embeddings, labels) if n_clusters > 1 else np.nan,
            'cluster_stats': self._get_cluster_distribution(labels),
            'stability': self.calculate_stability(embeddings, labels)
        }
    
    # 新增稳定性分析
    def calculate_stability(self, embeddings, labels, n_runs=5):
        original_labels = labels.copy()
        scores = []
        
        for _ in range(n_runs):
            temp_labels = KMeans(n_clusters=len(np.unique(original_labels))).fit_predict(embeddings)
            score = adjusted_rand_score(original_labels, temp_labels)
            scores.append(score)
            
        return {
            'mean_ari': np.mean(scores),
            'std_ari': np.std(scores)
        }

    # 新增缺失的方法
    def _get_cluster_distribution(self, labels):
        """获取簇分布统计"""
        counts = Counter(labels)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return {
            'distribution': dict(sorted_counts),
            'avg_size': np.mean([c for k,c in counts.items() if k != -1]),
            'max_size': max([c for k,c in counts.items() if k != -1], default=0),
            'min_size': min([c for k,c in counts.items() if k != -1], default=0)
        }

    # 新增对比可视化
    def plot_comparison(self, results, save_path=None):
        methods = list(results.keys())
        metrics = ['time', 'silhouette', 'calinski_harabasz']
        
        plt.figure(figsize=(15, 5))
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, idx)
            values = [results[m][metric] for m in methods]
            plt.bar(methods, values)
            plt.title(metric.upper())
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # 新增降维可视化    
    def visualize_clusters(self, embeddings, labels_dict, save_path=None):
        tsne = TSNE(n_components=2, perplexity=self.config.COMPARISON_CONFIG['tsne_perplexity'])
        reduced = tsne.fit_transform(embeddings)
        
        num_methods = len(labels_dict)
        rows = (num_methods + 1) // 2
        cols = 2 if num_methods > 1 else 1
        
        plt.figure(figsize=(20, 15))
        for idx, (method, labels) in enumerate(labels_dict.items(), 1):
            plt.subplot(rows, cols, idx)
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10)
            plt.title(f"{method.upper()} Clustering")
            plt.colorbar(scatter)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    def calculate_metrics(self, embeddings, labels):
        """自动计算评估指标"""
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {'silhouette': np.nan, 'davies_bouldin': np.nan}
            
        return {
            'silhouette': silhouette_score(embeddings, labels),
            'davies_bouldin': davies_bouldin_score(embeddings, labels)
        }
    
    def get_cluster_keywords(self, documents, labels, top_n=None):
        """自动提取关键词"""
        top_n = top_n or self.config.TOP_KEYWORDS
        vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        tfidf = vectorizer.fit_transform(documents)
        
        keywords_dict = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
                
            cluster_mask = (labels == cluster_id)
            cluster_tfidf = tfidf[cluster_mask].mean(axis=0)
            feature_names = vectorizer.get_feature_names_out()
            
            top_indices = np.argsort(cluster_tfidf.A[0])[-top_n:][::-1]
            keywords_dict[cluster_id] = [feature_names[i] for i in top_indices]
            
        return keywords_dict