"""
Text Preprocessing Utilities

This module provides text preprocessing functions for the Neural Retriever-Reranker
RAG pipeline, including text cleaning, normalization, and feature extraction.
"""

import re
import string
from typing import List, Dict, Any, Optional, Union
import logging

import pandas as pd
import numpy as np

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class TextPreprocessor:
    """
    Comprehensive text preprocessor for document and query processing.
    
    Supports various preprocessing options including tokenization, normalization,
    stopword removal, stemming, and lemmatization.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        expand_contractions: bool = True,
        remove_digits: bool = False,
        lemmatize: bool = False,
        stem: bool = False,
        min_token_length: int = 2,
        language: str = 'english'
    ):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_special_chars: Remove special characters
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove stopwords
            expand_contractions: Expand contractions (e.g., don't -> do not)
            remove_digits: Remove numeric digits
            lemmatize: Apply lemmatization
            stem: Apply stemming (mutually exclusive with lemmatization)
            min_token_length: Minimum token length to keep
            language: Language for stopwords and stemming
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.expand_contractions = expand_contractions
        self.remove_digits = remove_digits
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_token_length = min_token_length
        self.language = language
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            self._setup_nltk()
            
            if remove_stopwords:
                try:
                    self.stopwords = set(stopwords.words(language))
                except OSError:
                    self.logger.warning(f"Stopwords for {language} not available")
                    self.stopwords = set()
            else:
                self.stopwords = set()
            
            if stem:
                self.stemmer = PorterStemmer()
            else:
                self.stemmer = None
                
            if lemmatize:
                self.lemmatizer = WordNetLemmatizer()
            else:
                self.lemmatizer = None
        else:
            self.logger.warning("NLTK not available. Limited preprocessing options.")
            self.stopwords = set()
            self.stemmer = None
            self.lemmatizer = None
        
        # Contraction mapping
        self.contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "let's": "let us",
            "that's": "that is", "who's": "who is", "what's": "what is",
            "here's": "here is", "there's": "there is", "where's": "where is",
            "it's": "it is", "he's": "he is", "she's": "she is"
        }
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for dataset in required_data:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else 
                              f'corpora/{dataset}' if dataset in ['stopwords', 'wordnet'] else
                              f'taggers/{dataset}')
            except LookupError:
                try:
                    nltk.download(dataset, quiet=True)
                except:
                    self.logger.warning(f"Could not download NLTK data: {dataset}")
    
    def process_text(self, text: str) -> str:
        """
        Process a single text string through all preprocessing steps.
        
        Args:
            text: Input text string
            
        Returns:
            Processed text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return ""
        
        # Apply preprocessing steps in order
        if self.lowercase:
            text = text.lower()
        
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
        
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        if self.remove_digits:
            text = self._remove_digits(text)
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Apply token-level processing
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)
        
        if self.stem and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.lemmatize and self.lemmatizer:
            tokens = self._lemmatize_tokens(tokens)
        
        # Filter by length
        tokens = [token for token in tokens if len(token) >= self.min_token_length]
        
        return " ".join(tokens)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and spaces."""
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_digits(self, text: str) -> str:
        """Remove numeric digits from text."""
        return re.sub(r'\d+', '', text)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            return word_tokenize(text)
        else:
            # Simple whitespace tokenization
            return text.split()
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens."""
        if not self.lemmatizer:
            return tokens
        
        # Get POS tags for better lemmatization
        try:
            pos_tags = pos_tag(tokens)
            lemmatized = []
            
            for token, pos in pos_tags:
                # Convert POS tag to WordNet format
                wordnet_pos = self._get_wordnet_pos(pos)
                lemmatized.append(self.lemmatizer.lemmatize(token, wordnet_pos))
            
            return lemmatized
        except:
            # Fallback to simple lemmatization
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert Treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
    
    def process_batch(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            List of processed texts
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Processing texts")
            except ImportError:
                iterator = texts
        else:
            iterator = texts
        
        return [self.process_text(text) for text in iterator]
    
    def get_config(self) -> Dict[str, Any]:
        """Get current preprocessing configuration."""
        return {
            'lowercase': self.lowercase,
            'remove_special_chars': self.remove_special_chars,
            'remove_punctuation': self.remove_punctuation,
            'remove_stopwords': self.remove_stopwords,
            'expand_contractions': self.expand_contractions,
            'remove_digits': self.remove_digits,
            'lemmatize': self.lemmatize,
            'stem': self.stem,
            'min_token_length': self.min_token_length,
            'language': self.language
        }


def preprocess_node_text(
    node_data: Union[pd.DataFrame, Dict[str, Any]], 
    text_fields: List[str],
    preprocessor: Optional[TextPreprocessor] = None
) -> Union[List[str], str]:
    """
    Preprocess text fields from node data.
    
    Args:
        node_data: Node data (DataFrame or dictionary)
        text_fields: List of text field names to process
        preprocessor: Text preprocessor instance (creates default if None)
        
    Returns:
        Processed text(s)
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    if isinstance(node_data, pd.DataFrame):
        # Process DataFrame
        processed_texts = []
        
        for _, row in node_data.iterrows():
            combined_text = combine_text_fields(row, text_fields)
            processed_text = preprocessor.process_text(combined_text)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    else:
        # Process single record
        combined_text = combine_text_fields(node_data, text_fields)
        return preprocessor.process_text(combined_text)


def combine_text_fields(
    record: Union[pd.Series, Dict[str, Any]], 
    text_fields: List[str],
    field_separators: bool = True
) -> str:
    """
    Combine multiple text fields into a single text string.
    
    Args:
        record: Data record (Series or dictionary)
        text_fields: List of field names to combine
        field_separators: Whether to add field name separators
        
    Returns:
        Combined text string
    """
    text_parts = []
    
    for field in text_fields:
        if field in record and pd.notna(record[field]):
            value = record[field]
            
            # Process different data types
            if isinstance(value, str) and value.strip():
                if field_separators:
                    text_parts.append(f"{field}: {value.strip()}")
                else:
                    text_parts.append(value.strip())
                    
            elif isinstance(value, list) and value:
                # Handle list fields (features, reviews, etc.)
                list_text = process_list_field(value)
                if list_text:
                    if field_separators:
                        text_parts.append(f"{field}: {list_text}")
                    else:
                        text_parts.append(list_text)
                        
            elif value is not None:
                # Convert other types to string
                str_value = str(value).strip()
                if str_value and str_value.lower() not in ['nan', 'none', 'null']:
                    if field_separators:
                        text_parts.append(f"{field}: {str_value}")
                    else:
                        text_parts.append(str_value)
    
    return " ".join(text_parts)


def process_list_field(field_value: List[Any]) -> str:
    """
    Process list-type fields (reviews, features, etc.) into text.
    
    Args:
        field_value: List of items (strings, dictionaries, etc.)
        
    Returns:
        Combined text from list items
    """
    if not field_value:
        return ""
    
    text_items = []
    
    for item in field_value:
        if isinstance(item, str):
            text_items.append(item.strip())
        elif isinstance(item, dict):
            # Handle dictionary items (like reviews with multiple fields)
            dict_text = process_dict_field(item)
            if dict_text:
                text_items.append(dict_text)
        else:
            # Convert other types to string
            str_item = str(item).strip()
            if str_item and str_item.lower() not in ['nan', 'none', 'null']:
                text_items.append(str_item)
    
    return " ".join(text_items)


def process_dict_field(field_value: Dict[str, Any]) -> str:
    """
    Process dictionary-type fields into text.
    
    Args:
        field_value: Dictionary with text content
        
    Returns:
        Combined text from dictionary values
    """
    if not field_value:
        return ""
    
    # Priority fields for different types of dictionaries
    priority_fields = ['text', 'content', 'summary', 'reviewText', 'title', 'description']
    
    text_parts = []
    
    # First, try priority fields
    for field in priority_fields:
        if field in field_value and field_value[field]:
            value = str(field_value[field]).strip()
            if value and value.lower() not in ['nan', 'none', 'null']:
                text_parts.append(value)
                break  # Use only the first priority field found
    
    # If no priority field found, use all text fields
    if not text_parts:
        for key, value in field_value.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
            elif value is not None:
                str_value = str(value).strip()
                if str_value and str_value.lower() not in ['nan', 'none', 'null']:
                    text_parts.append(str_value)
    
    return " ".join(text_parts)


def clean_amazon_fields(node_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Amazon-specific fields with known formatting issues.
    
    Args:
        node_data: DataFrame with Amazon product data
        
    Returns:
        DataFrame with cleaned fields
    """
    df = node_data.copy()
    
    # Clean price field
    if 'price' in df.columns:
        df['price'] = df['price'].apply(clean_price_field)
    
    # Clean rank field
    if 'rank' in df.columns:
        df['rank'] = df['rank'].apply(clean_rank_field)
    
    # Clean category fields
    for cat_field in ['category', 'global_category']:
        if cat_field in df.columns:
            df[cat_field] = df[cat_field].apply(clean_category_field)
    
    # Process review fields
    if 'review' in df.columns:
        df['review'] = df['review'].apply(process_review_field)
    
    # Process features field
    if 'feature' in df.columns:
        df['feature'] = df['feature'].apply(process_feature_field)
    
    return df


def clean_price_field(price_value: Any) -> str:
    """Clean price field by removing currency symbols and formatting."""
    if pd.isna(price_value) or price_value in ['', 'blank']:
        return ""
    
    price_str = str(price_value)
    # Remove common currency symbols and extra spaces
    price_str = re.sub(r'[$£€¥₹]', '', price_str)
    price_str = re.sub(r'\s+', ' ', price_str).strip()
    
    return price_str


def clean_rank_field(rank_value: Any) -> str:
    """Clean rank field by extracting ranking information."""
    if pd.isna(rank_value):
        return ""
    
    rank_str = str(rank_value)
    # Extract ranking number and category
    # Example: "950,041 in Sports & Outdoors" -> "950041 Sports Outdoors"
    rank_str = re.sub(r'[,#]', '', rank_str)  # Remove commas and hash symbols
    rank_str = re.sub(r'\s+in\s+', ' ', rank_str)  # Replace " in " with space
    rank_str = re.sub(r'[&]', '', rank_str)  # Remove ampersands
    rank_str = re.sub(r'\s+', ' ', rank_str).strip()
    
    return rank_str


def clean_category_field(category_value: Any) -> str:
    """Clean category field."""
    if pd.isna(category_value):
        return ""
    
    if isinstance(category_value, list):
        return " ".join(str(cat).strip() for cat in category_value if str(cat).strip())
    else:
        return str(category_value).strip()


def process_review_field(reviews: Any) -> str:
    """Process reviews field to extract meaningful text."""
    if pd.isna(reviews) or not reviews:
        return ""
    
    if isinstance(reviews, str):
        try:
            # Try to parse as JSON/eval if it's a string representation
            import ast
            reviews = ast.literal_eval(reviews)
        except:
            return reviews  # Return as-is if parsing fails
    
    if isinstance(reviews, list):
        review_texts = []
        for review in reviews[:5]:  # Limit to first 5 reviews
            if isinstance(review, dict):
                # Extract review text from dictionary
                text = ""
                for field in ['reviewText', 'summary', 'text', 'content']:
                    if field in review and review[field]:
                        text = str(review[field]).strip()
                        break
                if text:
                    review_texts.append(text)
            elif isinstance(review, str):
                review_texts.append(review.strip())
        
        return " ".join(review_texts)
    
    return str(reviews)


def process_feature_field(features: Any) -> str:
    """Process features field to extract feature text."""
    if pd.isna(features) or not features:
        return ""
    
    if isinstance(features, str):
        try:
            import ast
            features = ast.literal_eval(features)
        except:
            return features
    
    if isinstance(features, list):
        feature_texts = []
        for feature in features:
            if isinstance(feature, str) and feature.strip():
                feature_texts.append(feature.strip())
            elif feature is not None:
                feature_str = str(feature).strip()
                if feature_str:
                    feature_texts.append(feature_str)
        
        return " ".join(feature_texts)
    
    return str(features)


def calculate_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Calculate statistics about processed texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with text statistics
    """
    if not texts:
        return {}
    
    # Filter out empty texts
    non_empty_texts = [text for text in texts if text.strip()]
    
    if not non_empty_texts:
        return {"empty_texts": len(texts)}
    
    # Calculate various statistics
    lengths = [len(text) for text in non_empty_texts]
    word_counts = [len(text.split()) for text in non_empty_texts]
    
    stats = {
        'total_texts': len(texts),
        'non_empty_texts': len(non_empty_texts),
        'empty_texts': len(texts) - len(non_empty_texts),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'avg_words': np.mean(word_counts),
        'median_words': np.median(word_counts),
        'min_words': np.min(word_counts),
        'max_words': np.max(word_counts)
    }
    
    return stats


def create_combined_text(
    record: Union[pd.Series, Dict], 
    fields: List[str],
    preprocessor: Optional[TextPreprocessor] = None
) -> str:
    """
    Create combined and optionally preprocessed text from record fields.
    
    Args:
        record: Data record
        fields: List of field names to combine
        preprocessor: Optional text preprocessor
        
    Returns:
        Combined (and optionally processed) text
    """
    # Combine fields
    combined = combine_text_fields(record, fields, field_separators=True)
    
    # Apply preprocessing if provided
    if preprocessor:
        combined = preprocessor.process_text(combined)
    
    return combined


# Predefined preprocessing configurations
PREPROCESSING_CONFIGS = {
    'minimal': {
        'lowercase': True,
        'remove_special_chars': False,
        'remove_punctuation': False,
        'remove_stopwords': False,
        'expand_contractions': True,
        'remove_digits': False,
        'lemmatize': False,
        'stem': False
    },
    'standard': {
        'lowercase': True,
        'remove_special_chars': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'expand_contractions': True,
        'remove_digits': False,
        'lemmatize': False,
        'stem': False
    },
    'aggressive': {
        'lowercase': True,
        'remove_special_chars': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'expand_contractions': True,
        'remove_digits': True,
        'lemmatize': True,
        'stem': False
    },
    'bm25_optimized': {
        'lowercase': True,
        'remove_special_chars': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'expand_contractions': True,
        'remove_digits': False,
        'lemmatize': False,
        'stem': False
    }
}


def get_preprocessor(config_name: str = 'standard') -> TextPreprocessor:
    """
    Get a preconfigured text preprocessor.
    
    Args:
        config_name: Name of preprocessing configuration
        
    Returns:
        Configured TextPreprocessor instance
    """
    if config_name not in PREPROCESSING_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(PREPROCESSING_CONFIGS.keys())}")
    
    config = PREPROCESSING_CONFIGS[config_name]
    return TextPreprocessor(**config)