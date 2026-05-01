import math
from collections import defaultdict

import spacy
import yake
from nltk.corpus import wordnet


# CONFIGURATION
COLLECTION_FILE  = "CISI.ALLnettoye"
QUERIES_FILE     = "CISI_dev.QRY"
RELEVANCE_FILE   = "CISI_dev.REL"
OUTPUT_FILE      = "results.txt"
TOP_K            = 100

# Paramètres BM25
BM25_K1 = 1.5
BM25_B  = 0.75

# Paramètres Rocchio
ROCCHIO_ALPHA    = 1.0
ROCCHIO_BETA     = 0.75
ROCCHIO_TOP_DOCS = 5

# Paramètres NLP
SYNONYM_MAX  = 2    # nb max de synonymes WordNet par terme de requête
YAKE_BOOST   = 1.5  # facteur de boost pour les termes clés Yake dans BM25
YAKE_TOP     = 5    # nb de termes clés extraits par Yake par requête


# CHARGEMENT DES OUTILS NLP
nlp            = spacy.load("en_core_web_sm", disable=["parser", "ner"])
yake_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=YAKE_TOP)

# ÉTAPE 1 : TOKENISATION & LEMMATISATION (spaCy)
def tokenize(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if (not token.is_stop and
                not token.is_punct and
                not token.is_space and
                token.is_alpha and
                len(token.lemma_) >= 3 and
                token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}):
            tokens.append(token.lemma_)
    return tokens

# ÉTAPE 1b : EXPANSION SYNONYMES (WordNet)
def expand_with_synonyms(tokens):
    expanded = list(tokens)
    seen = set(tokens)
    for token in tokens:
        count = 0
        for syn in wordnet.synsets(token)[:2]:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                if (synonym != token and
                        synonym.isalpha() and
                        len(synonym) >= 3 and
                        synonym not in seen and
                        count < SYNONYM_MAX):
                    expanded.append(synonym)
                    seen.add(synonym)
                    count += 1
    return expanded

# ÉTAPE 1c : EXTRACTION TERMES CLÉS (Yake)
def get_key_terms(text):
    keywords = yake_extractor.extract_keywords(text)
    return {kw.lower() for kw, _ in keywords}


# PARSING DES FICHIERS
def parse_collection(filepath):
    docs = {}
    current_id = None
    current_text = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith(".I "):
                if current_id is not None:
                    docs[current_id] = " ".join(current_text).strip()
                current_id = int(line[3:].strip())
                current_text = []
            elif line.startswith("."):
                pass
            else:
                current_text.append(line.strip())
    if current_id is not None:
        docs[current_id] = " ".join(current_text).strip()
    print(f"✓ Collection chargée : {len(docs)} documents")
    return docs




if __name__ == "__main__":
    main()
