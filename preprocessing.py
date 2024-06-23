import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_documents(documents):
    def preprocess(text):
        try:
            tokens = [word.lower() for word in text.split() if word.lower() not in stop_words]
            return tokens
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return []
    
    return [preprocess(doc) if isinstance(doc, str) else [] for doc in documents]