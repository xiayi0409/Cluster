# %% utils/preprocess.py
import os
import re
from markdown import markdown
from bs4 import BeautifulSoup

def load_and_preprocess_md(folder_path):
    """
    加载并预处理Markdown文件
    返回: 处理后的纯文本列表
    """
    texts = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.md'):
            continue
            
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()
            
        # 转换Markdown为纯文本
        html = markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # 清洗文本
        text = re.sub(r'\s+', ' ', text)          # 合并空白字符
        text = re.sub(r'[^\w\s-]', '', text)      # 移除标点
        text = text.lower().strip()
        
        texts.append(text)
    return texts