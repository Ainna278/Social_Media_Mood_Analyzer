import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ── Label Mapping ──────────────────────────────────────────

emotion_to_mood = {
    0: 'happy',   # admiration
    1: 'happy',   # amusement
    2: 'angry',   # anger
    3: 'angry',   # annoyance
    4: 'happy',   # approval
    5: 'happy',   # caring
    6: 'neutral', # confusion
    7: 'neutral', # curiosity
    8: 'happy',   # desire
    9: 'sad',     # disappointment
    10: 'angry',  # disapproval
    11: 'angry',  # disgust
    12: 'sad',    # embarrassment
    13: 'happy',  # excitement
    14: 'sad',    # fear
    15: 'happy',  # gratitude
    16: 'sad',    # grief
    17: 'happy',  # joy
    18: 'happy',  # love
    19: 'sad',    # nervousness
    20: 'happy',  # optimism
    21: 'happy',  # pride
    22: 'neutral',# realization
    23: 'happy',  # relief
    24: 'sad',    # remorse
    25: 'sad',    # sadness
    26: 'neutral',# surprise
    27: 'neutral' # neutral
}

mood_priority = {
    'angry': 0,
    'sad': 1,
    'happy': 2,
    'neutral': 3
}

def assign_mood(labels):
    """
    Convert list of emotion labels to single mood
    using priority rule.

    Args:
        labels: list of emotion indices
    Returns:
        single mood string
    """
    moods = [emotion_to_mood[label] for label in labels]
    return min(moods, key=lambda mood: mood_priority[mood])


# ── Preprocessing Pipeline ─────────────────────────────────

def remove_noise(text):
    """Remove URLs, emojis, hashtags, and special symbols."""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    """Normalize informal English words to standard form."""
    normalization_dict = {
        "u": "you",
        "ur": "your",
        "r": "are",
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "cant": "cannot",
        "wont": "will not",
        "ok": "okay",
        "omg": "oh my god",
        "lol": "laughing",
        "wtf": "what the",
        "idk": "i do not know",
        "imo": "in my opinion",
        "tbh": "to be honest",
        "ngl": "not going to lie",
        "abt": "about",
        "bc": "because",
        "b4": "before",
        "gr8": "great",
        "luv": "love",
        "msg": "message",
        "pls": "please",
        "thx": "thanks",
        "yr": "your",
        "irl": "in real life",
        "fyi": "for your information",
        "afaik": "as far as i know",
    }
    words = text.split()
    words = [normalization_dict.get(word.lower(), word) for word in words]
    return ' '.join(words)

def lowercase_text(text):
    """Convert all text to lowercase."""
    return text.lower()

def remove_punctuation(text):
    """Remove punctuation from text."""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """Split text into individual word tokens."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Remove common English stopwords."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def get_wordnet_pos(tag):
    """Convert NLTK POS tag to WordNet POS tag."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens):
    """Reduce words to root form using POS aware lemmatization."""
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in pos_tags]

def preprocess(text):
    """
    Full preprocessing pipeline.

    Args:
        text: raw social media text
    Returns:
        cleaned preprocessed string
    """
    text = remove_noise(text)
    text = normalize_text(text)
    text = lowercase_text(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)