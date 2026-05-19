# src/preprocess.py

# Emotion to mood mapping
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

# Priority rule - lower number = higher priority
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