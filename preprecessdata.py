import re
import nltk
from nltk.corpus import stopwords

# 下载停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('chinese'))

def clean_text(text):
    # 去掉HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去掉非中文字符
    text = re.sub(r'[^a-zA-Z\u4e00-\u9fa5]', ' ', text)
    # 分词
    words = text.split()
    # 去除停用词
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

cleaned_data = [clean_text(article['content']) for article in news_data]
print(cleaned_data[:5])
