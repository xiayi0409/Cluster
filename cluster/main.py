# main.py
import time
from config.settings import Settings
from data_loader.loader import DocumentLoader
from embedding.generator import EmbeddingGenerator
from clustering.core import ClusterEngine
from clustering.postprocess import ClusterPostProcessor
from evaluation.metrics import ClusterEvaluator
from sklearn.metrics import adjusted_rand_score  # 新增导入
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

class EnhancedClusteringPipeline:
    def __init__(self):
        self.config = Settings()
        self.loader = DocumentLoader(self.config)
        self.embedder = EmbeddingGenerator(self.config)
        self.cluster_engine = ClusterEngine(self.config)
        self.post_processor = ClusterPostProcessor(self.config)
        self.evaluator = ClusterEvaluator(self.config)

    def run_comparative_analysis(self):
        # 数据加载和嵌入生成
        docs = self.loader.load_documents()
        embeddings = self.embedder.generate_embeddings(docs)

        # 执行所有聚类方法   
        results = {}
        for method in self.config.CLUSTER_METHODS:
            start_time = time.time()  # 调用 time 模块中的 time() 函数
            labels = self.cluster_engine.cluster(method, embeddings)
            labels = self.post_processor.merge_similar_clusters(embeddings, labels)
            labels = self.post_processor.handle_noise_points(embeddings, labels)
            elapsed_time =  time.time() - start_time
            metrics = self.evaluator.calculate_all_metrics(embeddings, labels, elapsed_time)
            results[method] = {
                'labels': labels,
                **metrics
            }

        # 保存对比分析图表
        comparison_save_path = self.config.COMPARISON_CONFIG.get('comparison_save_path', 'cluster_comparison.png')
        self.evaluator.plot_comparison(results, save_path=comparison_save_path)

        # 保存聚类结果可视化图表
        visualization_save_path = self.config.COMPARISON_CONFIG.get('visualization_save_path', 'cluster_visualization.png')
        self.evaluator.visualize_clusters(embeddings, {m: results[m]['labels'] for m in results}, save_path=visualization_save_path)

        # 打印详细报告
        self._print_comparative_report(results)

        # 输出每个簇的关键词和对应文档到文件
        self._save_cluster_keywords_and_docs(docs, results)

        return results

    def _print_comparative_report(self, results):
        print("\n=== 聚类方法综合对比报告 ===")
        print("{:<10} | {:<8} | {:<8} | {:<8} | {:<6} | {:<6}".format(
            'Method', 'Time(s)', 'Clusters', 'Noise%', 'Silh', 'CH'))

        for method, data in results.items():
            print("{:<10} | {:>8.2f} | {:>8} | {:>7.1%} | {:>6.2f} | {:>6.1f}".format(
                method,
                data['time'],
                data['n_clusters'],
                data['noise_ratio'],
                data['silhouette'],
                data['calinski_harabasz']
            ))

        # 打印稳定性分析
        print("\n=== 稳定性分析 ===")
        for method, data in results.items():
            print(f"{method.upper()} ARI稳定性: {data['stability']['mean_ari']:.3f} ± {data['stability']['std_ari']:.3f}")

    def _save_cluster_keywords_and_docs(self, docs, results):
        for method, data in results.items():
            file_name = f"{method}_cluster_results.txt"
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(f"=== {method.upper()} 聚类结果关键词和文档 ===\n")
                labels = data['labels']
                keywords_dict = self.evaluator.get_cluster_keywords(docs, labels)

                for cluster_id, keywords in keywords_dict.items():
                    f.write(f"\n簇 {cluster_id} 关键词: {', '.join(keywords)}\n")
                    cluster_doc_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                    cluster_docs = [docs[i] for i in cluster_doc_indices]
                    file_names = [os.path.basename(self.loader.config.DATA_PATH + '/' + f) for f in os.listdir(self.loader.config.DATA_PATH) if f.endswith('.md')]
                    cluster_file_names = [file_names[i] for i in cluster_doc_indices]
                    f.write(f"簇 {cluster_id} 文档数量: {len(cluster_docs)}\n")
                    f.write("簇 {cluster_id} 文档名字:\n")
                    for file_name in cluster_file_names:
                        f.write(f"  - {file_name}\n")
                    for idx, doc in enumerate(cluster_docs, start=1):
                        f.write(f"文档 {idx}: {doc[:200]}...\n")  # 仅打印文档前200个字符

if __name__ == "__main__":
    pipeline = EnhancedClusteringPipeline()
    results = pipeline.run_comparative_analysis()