"""
utils/preprocessing.py
Semua fungsi preprocessing teks: case folding, normalisasi,
cleaning, tokenisasi, stopword removal, lemmatisasi.
"""

import re
import string
import streamlit as st

# ── Kamus singkatan ────────────────────────────────────────────────────────────
ABBREVIATIONS = {
    "pls":"please","f2p":"free to play","tbf":"to be fair","idk":"i do not know",
    "tbh":"to be honest","omg":"oh my god","lol":"laugh out loud","btw":"by the way",
    "nvm":"nevermind","ofc":"of course","ppl":"people","rn":"right now",
    "rlly":"really","rly":"really","tho":"though","thx":"thanks","tmr":"tomorrow",
    "u":"you","ur":"you are","bc":"because","b4":"before","luv":"love",
    "gonna":"going to","gotta":"got to","wanna":"want to","kinda":"kind of",
    "can't":"cannot","won't":"will not","don't":"do not","doesn't":"does not",
    "didn't":"did not","isn't":"is not","aren't":"are not","wasn't":"was not",
    "i'm":"i am","i've":"i have","i'll":"i will","it's":"it is","that's":"that is",
    "he's":"he is","she's":"she is","they're":"they are","we're":"we are",
    "you're":"you are","let's":"let us","there's":"there is","what's":"what is",
    "sux":"sucks","gimme":"give me","lemme":"let me","ima":"i am going to",
}

# ── Pola regex (dikompilasi sekali) ────────────────────────────────────────────
URL_PAT   = re.compile(r'https?://\S+|www\.\S+')
NUM_PAT   = re.compile(r'\d+')
SPEC_PAT  = re.compile(r'<.*?>+')
PUNCT_PAT = re.compile(f'[{re.escape(string.punctuation)}]')
NL_PAT    = re.compile(r'\n')
QUOTE_PAT = re.compile(r'[\u2018\u2019\u201c\u201d\u2026]')
EMOJI_PAT = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


# ── NLTK helpers (di-cache Streamlit) ─────────────────────────────────────────
@st.cache_resource
def load_nltk():
    """Download resource NLTK yang diperlukan."""
    import nltk
    for pkg in ['punkt_tab', 'stopwords', 'wordnet', 'punkt']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True


@st.cache_resource
def get_stop_words():
    """Kembalikan set stopwords bahasa Inggris."""
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))


# ── Fungsi preprocessing ───────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Terapkan full preprocessing pipeline pada satu teks:
    lowercase → normalisasi singkatan → hapus URL/angka/emoji/tanda baca.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    for short, full in ABBREVIATIONS.items():
        t = re.sub(r'\b' + re.escape(short) + r'\b', full, t, flags=re.IGNORECASE)
    t = URL_PAT.sub('', t)
    t = NUM_PAT.sub('', t)
    t = SPEC_PAT.sub('', t)
    t = PUNCT_PAT.sub('', t)
    t = NL_PAT.sub(' ', t)
    t = QUOTE_PAT.sub('', t)
    t = EMOJI_PAT.sub('', t)
    return t.strip()


def tokenize_remove_stopwords(text: str, stop_words: set) -> list:
    """Tokenisasi dan hapus stopwords + kata < 3 karakter."""
    return [w for w in text.split() if w not in stop_words and len(w) > 2]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatisasi list token dengan WordNetLemmatizer (v → n → a)."""
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        result = []
        for w in tokens:
            lemma = lem.lemmatize(w, 'v')
            if lemma == w:
                lemma = lem.lemmatize(w, 'n')
            if lemma == w:
                lemma = lem.lemmatize(w, 'a')
            result.append(lemma)
        return result
    except Exception:
        return tokens


def full_preprocess(text: str, stop_words: set) -> str:
    """
    Shortcut: jalankan seluruh pipeline (preprocess → tokenize → lemmatize)
    dan kembalikan sebagai string siap vektorisasi.
    """
    cleaned = preprocess_text(text)
    tokens  = tokenize_remove_stopwords(cleaned, stop_words)
    tokens  = lemmatize_tokens(tokens)
    return ' '.join(tokens)
