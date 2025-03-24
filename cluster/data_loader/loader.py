import os
import re
from markdown import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm

class DocumentLoader:
    def __init__(self, config):
        self.config = config
        
    def _clean_text(self, text):
        """3GPP文档专用清洗"""
        # 保留技术术语的正则表达式
        tech_terms = r'\b(LTE|NR|5G|UE|gNB|PDCP|RLC|MAC|PHY)\b'
        matches = re.findall(tech_terms, text, flags=re.IGNORECASE)
        
        # 移除版本号等无关信息
        cleaned = re.sub(r'3GPP TS \d{2}\.\d{3} V\d+\.\d+\.\d+', '', text)
        # 保留找到的技术术语
        return ' '.join(matches + [cleaned])
    
    def load_documents(self):
        """完整加载并预处理所有文档"""
        documents = []
        for root, dirs, files in os.walk(self.config.DATA_PATH):
            for filename in tqdm(files, desc="Loading Documents"):
                if not filename.endswith('.md'):
                    continue

                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()[:self.config.MAX_DOC_LENGTH]  # 控制最大长度

                # 转换Markdown为纯文本
                html = markdown(content)
                text = BeautifulSoup(html, 'html.parser').get_text()
                documents.append(self._clean_text(text))
        return documents