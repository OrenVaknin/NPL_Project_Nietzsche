import spacy
from spacy.language import Language
import json
import numpy as np
from spacy.tokens import Doc
import re, os, pathlib
import nltk
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def read_core_text(path: str) -> str:
    txt = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    s = txt.find("*** START OF")
    e = txt.find("*** END OF")
    if s != -1 and e != -1 and e > s:
        txt = txt[s:e]
    return txt

def split_into_paragraphs(core: str) -> list[str]:
    # join hard wraps inside paragraphs, keep real blank lines
    core = re.sub(r'(?<!\n)\n(?!\n)', ' ', core)
    # split on blank lines, treat whitespace only lines as blank
    paras = [p.strip() for p in re.split(r'(?:\n[ \t]*){2,}', core) if p.strip()]
    return paras

def normalize_book_stem(filename: str) -> str:
    # remove numeric prefix like "2The Gay Science"
    stem = pathlib.Path(filename).stem
    stem = re.sub(r'^\d+', '', stem).strip()
    return stem  

def build_clean_paragraphs(raw_dir: str, clean_paras_dir: str) -> None:
    os.makedirs(clean_paras_dir, exist_ok=True)
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        book_stem = normalize_book_stem(fn)
        core = read_core_text(os.path.join(raw_dir, fn))
        paras = split_into_paragraphs(core)
        for i, p in enumerate(paras, 1):
            out = f"{book_stem} para_{i:05d}.txt"
            with open(os.path.join(clean_paras_dir, out), "w", encoding="utf-8") as f:
                f.write(p)

# adding sentence tokenization rule for apostrophes for spacy
@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    if len(doc) == 0:
        return doc
    predecessor_token = doc[0]
    for token in doc[1:-1]:
        if token.text == "'" and predecessor_token.text == ".":
            doc[token.i + 1].is_sent_start = False
        predecessor_token = token
    return doc

def save_paras_tokenized_to_sents(read_path: str, write_path: str, nlp) -> None:
    os.makedirs(write_path, exist_ok=True)
    txt_files = [f for f in sorted(os.listdir(read_path)) if f.lower().endswith(".txt")]
    texts, outs = [], []
    for filename in txt_files:
        with open(os.path.join(read_path, filename), "r", encoding="utf-8") as f:
            texts.append(f.read())
        outs.append(os.path.join(write_path, filename.replace(".txt", ".json")))

    for doc, out_fp in zip(nlp.pipe(texts, batch_size=128), outs):
        sents = [s.text.replace("\n", " ").strip() for s in doc.sents if s.text.strip()]
        with open(out_fp, "w", encoding="utf-8") as jf:
            json.dump(sents, jf)

def extract_book_from_unit_filename(filename: str) -> str:
    # remove extension
    base = filename.rsplit(".", 1)[0]
    # drop trailing " para_00001" or " chap_0001"
    base = re.split(r'\s+(?:para|chap(?:ter)?)_\d+$', base)[0]
    # drop optional year in parentheses at end
    base = re.sub(r'\s*\(\d{4}\)\s*$', '', base).strip()
    return f"{base}.txt"  # match keys in period_map


def label_units_to_periods(tokenized_units_path: str, output_path: str) -> None:
    period_map = {
        1: ["The Birth Of Tragedy.txt", "Untimely Meditations.txt"],
        2: ["Human All Too Human.txt", "The Dawn of Day.txt", "The Gay Science.txt"],
        3: ["Beyond Good And Evil.txt", "Thus Spake Zarathustra.txt"],            
        4: ["The Antichrist.txt", "The Twilight of the Idols.txt"],
    }
def label_units_to_periods3(tokenized_units_path: str, output_path: str) -> None:
    period_map = {
        1: ["The Birth Of Tragedy.txt", "Untimely Meditations.txt"],
        2: ["Human All Too Human.txt", "The Dawn of Day.txt", "The Gay Science.txt"],
        3: ["Beyond Good And Evil.txt", "Thus Spake Zarathustra.txt", "The Antichrist.txt", "The Twilight of the Idols.txt"],            
    }
    book_to_period = {bk: p for p, books in period_map.items() for bk in books}

    labels = []
    for filename in sorted(os.listdir(tokenized_units_path)):
        if not filename.lower().endswith(".json"):
            continue
        book_name = extract_book_from_unit_filename(filename)
        labels.append(book_to_period.get(book_name, 0))
    with open(output_path, "w", encoding="utf-8") as jf:
        json.dump(labels, jf)

def label_units_by_book(tokenized_units_path: str, output_path: str) -> None:
    labels = []
    for filename in sorted(os.listdir(tokenized_units_path)):
        if not filename.lower().endswith(".json"):
            continue
        base = filename.rsplit(".", 1)[0]
        book = re.split(r'\s+(?:para|chap(?:ter)?)_\d+$', base)[0]
        book = re.sub(r'\s*\(\d{4}\)\s*$', '', book).strip()
        labels.append(book)
    with open(output_path, "w", encoding="utf-8") as jf:
        json.dump(labels, jf)

def paras_pos_percentage(doc, pos):
    num_of_pos = sum(1 for t in doc if t.pos_ == pos)
    num_of_words = sum(1 for t in doc if not t.is_punct and not t.is_space)
    return (num_of_pos / num_of_words) if num_of_words else 0.0
def paras_punctuation_percentage(x, nlp=None):
    if isinstance(x, Doc):
        doc = x
    else:
        paras_sentences = x
        text = " ".join(s for s in paras_sentences if s and s.strip())
        if not text or nlp is None:
            return 0.0
        doc = nlp(text)

    num_words = sum(1 for t in doc if not t.is_punct and not t.is_space)
    num_punct = sum(1 for t in doc if t.is_punct)
    return (num_punct / num_words) if num_words else 0.0

def build_pos_percentage_matrix(tokenized_units_path: str, nlp, out_path: str) -> None:
    files = sorted(f for f in os.listdir(tokenized_units_path) if f.lower().endswith(".json"))
    rows = []
    for fn in files:
        with open(os.path.join(tokenized_units_path, fn), "r", encoding="utf-8") as f:
            sents = json.load(f)
        text = " ".join(s for s in sents if s and s.strip())
        doc = nlp(text)
        row = [
            paras_pos_percentage(doc, "VERB"),
            paras_pos_percentage(doc, "ADJ"),
            paras_pos_percentage(doc, "ADV"),
            paras_pos_percentage(doc, "PROPN"),
            paras_pos_percentage(doc, "DET"),
            paras_pos_percentage(doc, "ADP"),
            paras_pos_percentage(doc, "PRON"),
            paras_punctuation_percentage(doc),  
            paras_pos_percentage(doc, "NOUN"),
        ]
        rows.append(row)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)

def sentiment_analysis(text):
    return vader_analyzer.polarity_scores(text)['compound']

def verbs_tenses_percent(sentences, nlp):
    past = present = future = 0
    for doc in nlp.pipe([s for s in sentences if s and s.strip()], batch_size=128):
        for t in doc:
            if t.pos_ in ("VERB","AUX"):
                tense = t.morph.get("Tense")
                if "Past" in tense: past += 1
                elif "Pres" in tense: present += 1
                elif "Fut" in tense: future += 1
    total = past + present + future
    return [x/(total or 1) for x in (past, present, future)]

def named_entity_percentage(paras_sentences, nlp):
    text = " ".join(s for s in paras_sentences if s and s.strip())
    if not text:
        return 0.0
    doc = nlp(text)
    num_words = sum(1 for t in doc if not t.is_punct and not t.is_space)
    num_entities = sum(1 for _ in doc.ents)  
    return (num_entities / num_words) if num_words else 0.0

vader_analyzer = SentimentIntensityAnalyzer()

FUNCTION_WORDS = [
    "the", "and", "but", "of", "to", "in", "a", "is", "that", "it", "with", "as", "for", "his", "he", "be", "was", "not", "by", "this", "they", "from", "or", "have", "an", "which", "one", "you", "had", "her", "were", "all", "she", "there", "would", "their", "we", "him", "been", "has", "when", "who", "will", "more", "no", "if", "out", "so", "said", "what", "up", "its", "about", "into", "than", "them", "can", "only", "other", "new", "some", "could", "time", "these", "two", "may", "then", "do", "first", "any", "my", "now", "such", "like", "our", "over", "man", "me", "even", "most", "made", "after", "also", "did", "many", "before", "must", "through", "back", "years", "where", "much", "your", "way", "well", "down", "should", "because", "each", "those", "people", "mr", "very", "after", "make", "through", "still", "take", "every", "here", "just", "something", "think", "know", "little", "too", "under", "own", "life", "day", "might", "part", "against", "go", "place", "around", "however", "seem", "another", "call", "why", "ask", "work", "world", "high", "different", "company", "hand", "off", "play", "turn", "study", "again", "animal", "point", "mother", "young", "home", "light", "country", "father", "let", "night", "picture", "being", "example", "paper", "group", "always", "music", "those", "both", "mark", "often", "letter", "until", "mile", "river", "car", "feet", "care", "second", "book", "carry", "took", "science", "eat", "room", "friend", "began", "idea", "fish", "mountain", "stop", "once", "base", "hear", "horse", "cut", "sure", "watch", "color", "face", "wood", "main", "enough", "plain", "girl", "usual", "young", "ready", "above", "ever", "red", "list", "though", "feel", "talk", "bird", "soon", "body", "dog", "family", "direct", "pose", "leave", "song", "measure", "door", "product", "black", "short", "numeral", "class", "wind", "question", "happen", "complete", "ship", "area", "half", "rock", "order", "fire", "south", "problem", "piece", "told", "knew", "pass", "since", "top", "whole", "king", "space", "heard", "best", "hour", "better", "true", "during", "hundred", "five", "remember", "step", "early", "hold", "west", "ground", "interest", "reach", "fast", "verb", "sing", "listen", "six", "table", "travel", "less", "morning", "ten", "simple", "several", "vowel", "toward", "war", "lay", "against", "pattern", "slow", "center", "love", "person", "money", "serve", "appear", "road", "map", "rain", "rule", "govern", "pull", "cold", "notice", "voice", "unit", "power", "town", "fine", "certain", "fly", "fall", "lead", "cry", "dark", "machine", "note", "wait", "plan", "figure", "star", "box", "noun", "field", "rest", "correct", "able", "pound", "done", "beauty", "drive", "stood", "contain", "front", "teach", "week", "final", "gave", "green", "oh", "quick", "develop", "ocean", "warm", "free", "minute", "strong", "special", "mind", "behind", "clear", "tail", "produce", "fact", "street", "inch", "multiply", "nothing", "course", "stay", "wheel", "full", "force", "blue", "object", "decide", "surface", "deep", "moon", "island", "foot", "system", "busy", "test", "record", "boat", "common", "gold", "possible", "plane", "stead", "dry", "wonder", "laugh", "thousand", "ago", "ran", "check", "game", "shape", "equate", "hot", "miss", "brought", "heat", "snow", "tire", "bring", "yes", "distant", "fill", "east", "paint", "language", "among", "grand", "ball", "yet", "wave", "drop", "heart", "am", "present", "heavy", "dance", "engine", "position", "arm", "wide", "sail", "material", "size", "vary", "settle", "speak", "weight", "general", "ice", "matter", "circle", "pair", "include", "divide", "syllable", "felt", "perhaps", "pick", "sudden", "count", "square", "reason", "length", "represent", "art", "subject", "region", "energy", "hunt", "probable", "bed", "brother", "egg", "ride", "cell", "believe", "fraction", "forest", "sit", "race", "window", "store", "summer", "train", "sleep", "prove", "lone", "leg", "exercise", "wall", "catch", "mount", "wish", "sky", "board", "joy", "winter", "sat", "written", "wild", "instrument", "kept", "glass", "grass", "cow", "job", "edge", "sign", "visit", "past", "soft", "fun", "bright", "gas", "weather", "month", "million", "bear", "finish", "happy", "hope", "flower", "clothe", "strange", "gone", "trade", "melody", "trip", "office", "receive", "row", "mouth", "exact", "symbol", "die", "least", "trouble", "shout", "except", "wrote", "seed", "tone", "join", "suggest", "clean", "break", "lady", "yard", "rise", "bad", "blow", "oil", "blood", "touch", "grew", "cent", "mix", "team", "wire", "cost", "lost", "brown", "wear", "garden", "equal", "sent", "choose", "fell", "fit", "flow", "fair", "bank", "collect", "save", "control", "decimal", "gentle", "woman", "captain", "practice", "separate", "difficult", "doctor", "please", "protect", "noon", "whose", "locate", "ring", "character", "insect", "caught", "period", "indicate", "radio", "spoke", "atom", "human", "history", "effect", "electric", "expect", "crop", "modern", "element", "hit", "student", "corner", "party", "supply", "bone", "rail", "imagine", "provide", "agree", "thus", "capital", "won't", "chair", "danger", "fruit", "rich", "thick", "soldier", "process", "operate", "guess", "necessary", "sharp", "wing", "create", "neighbor", "wash", "bat", "rather", "crowd", "corn", "compare", "poem", "string", "bell", "depend", "meat", "rub", "tube", "famous", "dollar", "stream", "fear", "sight", "thin", "triangle", "planet", "hurry", "chief", "colony", "clock", "mine", "tie", "enter", "major", "fresh", "search", "send", "yellow", "gun", "allow", "print", "dead", "spot", "desert", "suit", "current", "lift", "rose", "continue", "block", "chart", "hat", "sell", "success", "company", "subtract", "event", "particular", "deal", "swim", "term", "opposite", "wife", "shoe", "shoulder", "spread", "arrange", "camp", "invent", "cotton", "born", "determine", "quart", "nine", "truck", "noise", "level", "chance", "gather", "shop", "stretch", "throw", "shine", "property", "column", "molecule", "select", "wrong", "gray", "repeat", "require", "broad", "prepare", "salt", "nose", "plural", "anger", "claim", "continent", "oxygen", "sugar", "death", "pretty", "skill", "women", "season", "solution", "magnet", "silver", "thank", "branch", "match", "suffix", "especially", "fig", "afraid", "huge", "sister", "steel", "discuss", "forward", "similar", "guide", "experience", "score", "apple", "bought", "led", "pitch", "coat", "mass", "card", "band", "rope", "slip", "win", "dream", "evening", "condition", "feed", "tool", "total", "basic", "smell", "valley", "nor", "double", "seat", "arrive", "master", "track", "parent", "shore", "division", "sheet", "substance", "favor", "connect", "post", "spend", "chord", "fat", "glad", "original", "share", "station", "dad", "bread", "charge", "proper", "bar", "offer", "segment", "slave", "duck", "instant", "market", "degree", "populate", "chick", "dear", "enemy", "reply", "drink", "occur", "support", "speech", "nature", "range", "steam", "motion", "path", "liquid", "log", "meant", "quotient", "teeth", "shell", "neck"
]

def get_all_paragraph_files(tokenized_Paras_dir):
    """Get all paragraph files from the tokenized paragraphs directory."""
    paragraph_files = []
    for filename in os.listdir(tokenized_Paras_dir):
        if filename.endswith('.json'):
            paragraph_files.append(filename)
    return sorted(paragraph_files)

def load_paragraph_text(tokenized_Paras_dir, filename):
    """Load paragraph text from JSON file."""
    filepath = os.path.join(tokenized_Paras_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return ' '.join(data)

def calculate_readability_scores(text):
    """Calculate readability scores using textstat."""
    try:
        flesch_grade = textstat.flesch_kincaid_grade(text)
        dale_chall = textstat.dale_chall_readability_score(text)
        return flesch_grade, dale_chall
    except:
        return 0.0, 0.0

def calculate_lexical_diversity(doc):
    """Calculate Type-Token Ratio (TTR) using a spaCy Doc."""
    tokens = [t.text.lower() for t in doc if not t.is_space and not t.is_punct]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


FUNCTION_WORDS = list(dict.fromkeys([w.lower() for w in FUNCTION_WORDS]))

def calculate_function_word_profile(doc):
    """Normalized frequencies of FUNCTION_WORDS from a pre-parsed spaCy Doc."""
    tokens = [t.text.lower() for t in doc if not t.is_space and not t.is_punct]
    total = len(tokens) or 1
    counts = Counter(tokens)
    return [counts.get(w, 0) / total for w in FUNCTION_WORDS]

def calculate_sentiment_volatility(doc):
    """Calculate standard deviation of sentence-level sentiment scores."""
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if len(sentences) == 0:
        return 0.0
    
    sentiment_scores = []
    for sentence in sentences:
        if sentence.strip():
            scores = vader_analyzer.polarity_scores(sentence)
            sentiment_scores.append(scores['compound'])
    
    if len(sentiment_scores) == 0:
        return 0.0
    
    return np.std(sentiment_scores)

def extract_pos_bigrams(doc):
    """Return POS bigrams from a spaCy Doc, skipping spaces/punct."""
    pos_bigrams = []
    for i in range(len(doc) - 1):
        a, b = doc[i], doc[i+1]
        if a.is_space or a.is_punct or b.is_space or b.is_punct:
            continue
        pos_bigrams.append(f"{a.pos_}-{b.pos_}")
    return pos_bigrams


def get_top_pos_bigrams(tokenized_paras_dir, nlp, k=50):
    files = get_all_paragraph_files(tokenized_paras_dir)
    texts = [load_paragraph_text(tokenized_paras_dir, fn) for fn in files]
    counts = Counter()
    for doc in nlp.pipe(texts, batch_size=128):
        counts.update(extract_pos_bigrams(doc))
    return [bg for bg, _ in counts.most_common(k)]


def calculate_pos_bigram_frequencies(doc, top_bigrams):
    """Normalized frequency vector for top_k POS bigrams, using a Doc."""
    bigrams = extract_pos_bigrams(doc)
    counts = Counter(bigrams)
    total = sum(counts.values()) or 1
    return [counts.get(bg, 0) / total for bg in top_bigrams]


def extract_all_features(tokenized_paras_dir, textfeatures_dir, nlp):
    """Extract all 5 new features for all paragraphs."""
    print("Extracting new linguistic features...")
    
    paragraph_files = get_all_paragraph_files(tokenized_paras_dir)
    print(f"Found {len(paragraph_files)} paragraph files")
    
    print("Getting top POS bigrams...")
    top_bigrams = get_top_pos_bigrams(tokenized_paras_dir, nlp)
    
    readability_scores = []
    lexical_diversity_scores = []
    function_word_profiles = []
    sentiment_volatility_scores = []
    pos_bigram_frequencies = []
    
    for i, filename in enumerate(paragraph_files):
        if i % 50 == 0:
            print(f"Processing paragraph {i+1}/{len(paragraph_files)}")
        
        text = load_paragraph_text(tokenized_paras_dir, filename)
        doc = nlp(text)

        flesch_grade, dale_chall = calculate_readability_scores(text)
        readability_scores.append(flesch_grade) 

        lexical_diversity_scores.append(calculate_lexical_diversity(doc))

        function_word_profiles.append(calculate_function_word_profile(doc))

        sentiment_volatility_scores.append(calculate_sentiment_volatility(doc))

        pos_bigram_frequencies.append(calculate_pos_bigram_frequencies(doc, top_bigrams))

    print("Saving features to JSON files...")
    
    with open(os.path.join(textfeatures_dir, 'readability_scores.json'), 'w') as f:
        json.dump(readability_scores, f)
    with open(os.path.join(textfeatures_dir, 'lexical_diversity_scores.json'), 'w') as f:
        json.dump(lexical_diversity_scores, f)
    with open(os.path.join(textfeatures_dir, 'function_word_profiles.json'), 'w') as f:
        json.dump(function_word_profiles, f)
    with open(os.path.join(textfeatures_dir, 'sentiment_volatility_scores.json'), 'w') as f:
        json.dump(sentiment_volatility_scores, f)
    with open(os.path.join(textfeatures_dir, 'pos_bigram_frequencies.json'), 'w') as f:
        json.dump(pos_bigram_frequencies, f)
    with open(os.path.join(textfeatures_dir, 'top_pos_bigrams.json'), 'w') as f:
        json.dump(top_bigrams, f)
    
    print("Feature extraction completed!")
    print(f"Saved {len(readability_scores)} paragraphs worth of features")
    print(f"Function word profiles: {len(FUNCTION_WORDS)} words per paragraph")
    print(f"POS bigram frequencies: {len(top_bigrams)} bigrams per paragraph")

def create_full_feature_matrix(textfeatures_path):
    with open(os.path.join(textfeatures_path, 'paras_pos_percentage_matrix.json')) as f:
        pos = json.load(f)
    with open(os.path.join(textfeatures_path, 'Average_sentiment.json')) as f:
        sentiments = json.load(f)
    with open(os.path.join(textfeatures_path, 'tense_matrix.json')) as f:
        tense = json.load(f)
    with open(os.path.join(textfeatures_path, 'entities_matrix.json')) as f:
        entities = json.load(f)
    with open(os.path.join(textfeatures_path, 'punctuation_marks_matrix.json')) as f:
        punctuation = json.load(f)
    with open(os.path.join(textfeatures_path, 'readability_scores.json')) as f:
        readability = json.load(f)  
    with open(os.path.join(textfeatures_path, 'lexical_diversity_scores.json')) as f:
        ttr = json.load(f)          
    with open(os.path.join(textfeatures_path, 'function_word_profiles.json')) as f:
        func_words = json.load(f)   
    with open(os.path.join(textfeatures_path, 'sentiment_volatility_scores.json')) as f:
        sent_vol = json.load(f)     
    with open(os.path.join(textfeatures_path, 'pos_bigram_frequencies.json')) as f:
        pos_bigram_freqs = json.load(f)  
    n = len(pos)
    assert all(len(x) == n for x in [sentiments, tense, entities, punctuation,
                                     readability, ttr, func_words, sent_vol, pos_bigram_freqs])
    full_matrix = []
    for i in range(n):
        row = []
        row.extend(pos[i])
        row.append(sentiments[i])
        row.extend(tense[i])
        row.append(entities[i])
        row.append(punctuation[i])
        row.append(readability[i])       
        row.append(ttr[i])               
        row.extend(func_words[i])         
        row.extend(pos_bigram_freqs[i])   
        row.append(sent_vol[i])           
        full_matrix.append(row)
    with open(os.path.join(textfeatures_path, 'paras_full_feature_matrix.json'), 'w') as f:
        json.dump(full_matrix, f)

def main():
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('set_custom_boundaries', before="parser")


    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(ROOT_DIR, 'rawtxt')
    clean_paras_dir = os.path.join(ROOT_DIR, 'CleanParagraphs')
    tokenized_units_path = os.path.join(ROOT_DIR, 'Tokenized_Paras')
    textfeatures_path = os.path.join(ROOT_DIR, 'TextFeatures')
    os.makedirs(textfeatures_path, exist_ok=True)

    build_clean_paragraphs(raw_dir, clean_paras_dir)
    save_paras_tokenized_to_sents(clean_paras_dir, tokenized_units_path, nlp)

    label_units_to_periods3(
        tokenized_units_path,
        os.path.join(textfeatures_path, 'label_chaps_to_periods3.json')
    )
    label_units_by_book(
        tokenized_units_path,
        os.path.join(textfeatures_path, 'label_chaps_by_book.json')
    )


    files = sorted(os.listdir(tokenized_units_path))
    paragraphs = []
    for filename in files:
        with open(os.path.join(tokenized_units_path, filename), "r", encoding="utf-8") as f:
            paragraphs.append(" ".join(json.load(f)))
    average_sentiments = [sentiment_analysis(p) for p in paragraphs]
    with open(os.path.join(textfeatures_path, 'Average_sentiment.json'), 'w') as f:
        json.dump(average_sentiments, f)
    build_pos_percentage_matrix(
        tokenized_units_path,
        nlp,
        os.path.join(textfeatures_path, "paras_pos_percentage_matrix.json"),
    )

    tense_matrix, entities_matrix, punctuation_marks_matrix = [], [], []
    for filename in files:
        with open(os.path.join(tokenized_units_path, filename), "r", encoding="utf-8") as f:
            sents = json.load(f)
        tense_matrix.append(verbs_tenses_percent(sents, nlp))
        entities_matrix.append(named_entity_percentage(sents, nlp))
        punctuation_marks_matrix.append(paras_punctuation_percentage(sents, nlp))

    with open(os.path.join(textfeatures_path, 'tense_matrix.json'), 'w') as f:
        json.dump(tense_matrix, f)
    with open(os.path.join(textfeatures_path, 'entities_matrix.json'), 'w') as f:
        json.dump(entities_matrix, f)
    with open(os.path.join(textfeatures_path, 'punctuation_marks_matrix.json'), 'w') as f:
        json.dump(punctuation_marks_matrix, f)
    extract_all_features(tokenized_units_path, textfeatures_path, nlp)
    create_full_feature_matrix(textfeatures_path)
    print('All feature and label JSONs generated in TextFeatures/')

if __name__ == "__main__":
    main()
