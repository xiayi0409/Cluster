import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config.EMBED_MODEL_NAME)
        
    def _chunk_document(self, doc):
        """处理长文档的分块策略"""
        words = doc.split()
        chunks = []
        for i in range(0, len(words), self.config.CHUNK_SIZE - self.config.STRIDE):
            chunk = ' '.join(words[i:i+self.config.CHUNK_SIZE])
            chunks.append(chunk)
        return chunks
    
    def generate_embeddings(self, documents):
        """生成文档嵌入（完整内容处理）"""
        chunk_map = []  # 记录分块所属文档
        all_chunks = []
        
        # 分块处理
        for doc_idx, doc in enumerate(documents):
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            chunk_map.extend([doc_idx]*len(chunks))
            
        # 批量生成分块嵌入
        chunk_embeddings = self.model.encode(
            all_chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 合并为文档级嵌入
        doc_embeddings = np.zeros((len(documents), chunk_embeddings.shape[1]))
        for chunk_idx, doc_idx in enumerate(chunk_map):
            doc_embeddings[doc_idx] += chunk_embeddings[chunk_idx]
            
        # 平均池化
        doc_counts = np.bincount(chunk_map, minlength=len(documents))
        return doc_embeddings / doc_counts[:, np.newaxis]