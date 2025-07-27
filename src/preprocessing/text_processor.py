import re
import spacy
import nltk
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model {model_name} not found. Please install it using:")
            print(f"python -m spacy download {model_name}")
            self.nlp = None
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        if self.nlp:
            # Use spaCy for advanced processing
            doc = self.nlp(text)
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct and len(token.text) > 2]
        else:
            # Fallback to NLTK
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens 
                     if token.lower() not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text"""
        features = {}
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Basic features
            features['text_length'] = len(text)
            features['word_count'] = len(doc)
            features['sentence_count'] = len(list(doc.sents))
            
            # POS tag features
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            total_tokens = len(doc)
            for pos, count in pos_counts.items():
                features[f'pos_{pos.lower()}_ratio'] = count / total_tokens if total_tokens > 0 else 0
            
            # Named entities
            features['entity_count'] = len(doc.ents)
            
            # Sentiment (using spaCy's built-in sentiment if available)
            features['avg_token_length'] = np.mean([len(token.text) for token in doc]) if doc else 0
            
        return features
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 5000) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Create TF-IDF features"""
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        
        return tfidf_matrix.toarray(), self.tfidf_vectorizer
    
    def preprocess_for_ml(self, df: pd.DataFrame, text_column: str, target_column: str) -> Dict:
        """Complete preprocessing pipeline for ML"""
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Create TF-IDF features
        tfidf_features, vectorizer = self.create_tfidf_features(df['cleaned_text'].tolist())
        
        # Extract linguistic features
        linguistic_features = []
        for text in df['cleaned_text']:
            features = self.extract_features(text)
            linguistic_features.append(features)
        
        linguistic_df = pd.DataFrame(linguistic_features).fillna(0)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, linguistic_df.values])
        
        # Encode labels
        labels = self.label_encoder.fit_transform(df[target_column])
        
        return {
            'features': combined_features,
            'labels': labels,
            'feature_names': list(vectorizer.get_feature_names_out()) + list(linguistic_df.columns),
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': vectorizer,
            'linguistic_features': linguistic_df
        }
