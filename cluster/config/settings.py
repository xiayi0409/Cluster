class Settings:
    # 数据配置
    DATA_PATH = "/data2/wxy/cluster/data/Rel-18/"
    MAX_DOC_LENGTH = 9999999999999999  # 单个文档最大字符数

    # 嵌入模型配置
    EMBED_MODEL_NAME = "/data2/wxy/bge-large-en-v1.5/"
    CHUNK_SIZE = 2048  # 文本分块大小
    STRIDE = 128  # 分块重叠步长

    # 新增聚类方法配置（补充完整）
    CLUSTER_METHODS = {
        # K-Means
        'kmeans': {
            'n_clusters': 14,
            'random_state': 42
        },
        # DBSCAN
        'dbscan': {
            'eps': 0.5,  # 邻域半径
            'min_samples': 14  # 核心点最小样本数
        },
        # 层次聚类（AgglomerativeClustering）
        'hierarchical': {
            'n_clusters': 14,
            'linkage': 'ward'  # 链接方式（ward/average/complete）
        },
        # 高斯混合模型（GMM）
        'gmm': {
            'n_components': 14,  # 簇数量
            'random_state': 42
        },
        # HDBSCAN
        'hdbscan': {
            'min_cluster_size': 14,
            'cluster_selection_epsilon': 0.4
        },
        # 谱聚类（SpectralClustering）
        'spectral': {
            'n_clusters': 14,
            'affinity': 'nearest_neighbors'  # 相似度计算方式
        },
        # OPTICS
        'optics': {
            'min_samples': 14,
            'xi': 0.05  # 聚类显著性阈值
        }
    }

    # 新增对比分析配置
    COMPARISON_CONFIG = {
        'stability_runs': 5,  # 稳定性测试次数
        'top_keywords': 8,  # 显示关键词数量
        'tsne_perplexity': 30,  # 可视化参数
        'comparison_save_path': 'cluster_comparison.png',  # 对比分析图表保存路径
        'visualization_save_path': 'cluster_visualization.png'  # 聚类结果可视化图表保存路径
    }

    # 评估配置
    TOP_KEYWORDS = 40  # 每个簇显示的关键词数量