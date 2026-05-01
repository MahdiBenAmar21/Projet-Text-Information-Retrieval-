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

def parse_queries(filepath):
    queries = {}
    current_id = None
    current_text = []
    in_w_section = False
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith(".I "):
                if current_id is not None and current_text:
                    queries[current_id] = " ".join(current_text).strip()
                current_id = int(line[3:].strip())
                current_text = []
                in_w_section = False
            elif line.startswith(".W"):
                in_w_section = True
            elif line.startswith("."):
                in_w_section = False
            elif in_w_section:
                current_text.append(line.strip())
    if current_id is not None and current_text:
        queries[current_id] = " ".join(current_text).strip()
    queries = {k: v for k, v in queries.items() if k <= 30}
    print(f"✓ Requêtes chargées : {len(queries)} requêtes")
    return queries


# ÉTAPE 2 : INDEX TF-IDF (pour Rocchio)
def build_index(docs):
    N = len(docs)
    tokenized = {}
    print("  Tokenisation TF-IDF (spaCy)...")
    for doc_id, text in docs.items():
        tokenized[doc_id] = tokenize(text)

    tf = {}
    for doc_id, tokens in tokenized.items():
        counts = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        max_f = max(counts.values()) if counts else 1
        tf[doc_id] = {t: c / max_f for t, c in counts.items()}

    df = defaultdict(int)
    for term_freqs in tf.values():
        for term in term_freqs:
            df[term] += 1

    idf = {term: math.log(N / df_val) for term, df_val in df.items() if df_val > 0}

    tfidf = {
        doc_id: {term: tf_val * idf.get(term, 0) for term, tf_val in term_freqs.items()}
        for doc_id, term_freqs in tf.items()
    }
    print(f"✓ Index TF-IDF construit : {len(idf)} termes distincts")
    return tfidf, idf


# ÉTAPE 2b : INDEX BM25
def build_bm25(docs):
    N = len(docs)
    print("  Tokenisation BM25 (spaCy)...")
    tokenized = {doc_id: tokenize(text) for doc_id, text in docs.items()}

    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in tokenized.items()}
    avgdl = sum(doc_lengths.values()) / N if N > 0 else 1

    inverted = defaultdict(lambda: defaultdict(int))
    for doc_id, tokens in tokenized.items():
        for t in tokens:
            inverted[t][doc_id] += 1

    bm25_idf = {
        term: math.log((N - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
        for term, postings in inverted.items()
    }
    print(f"✓ Index BM25 construit : {len(bm25_idf)} termes, avgdl={avgdl:.1f}")
    return dict(inverted), bm25_idf, doc_lengths, avgdl


# ÉTAPE 3 : RECHERCHE BM25 + WordNet + Yake
def bm25_search(query_text, inverted, bm25_idf, doc_lengths, avgdl,
                top_k=TOP_K, k1=BM25_K1, b=BM25_B):
    # Tokenisation spaCy (même schéma que les documents)
    tokens = tokenize(query_text)
    if not tokens:
        return []

    # Expansion synonymes WordNet
    tokens = expand_with_synonyms(tokens)

    # Termes clés Yake pour boosting
    key_terms = get_key_terms(query_text)

    # TF normalisé de la requête (même schéma que les documents)
    q_counts = defaultdict(int)
    for t in tokens:
        q_counts[t] += 1
    max_f = max(q_counts.values())
    q_tf = {t: c / max_f for t, c in q_counts.items()}

    scores = defaultdict(float)
    for term, qtf in q_tf.items():
        if term not in inverted:
            continue
        idf_val = bm25_idf[term]
        # Boost Yake si le terme est un terme clé de la requête
        boost = YAKE_BOOST if term in key_terms else 1.0
        for doc_id, raw_tf in inverted[term].items():
            dl = doc_lengths[doc_id]
            num = raw_tf * (k1 + 1)
            den = raw_tf + k1 * (1 - b + b * dl / avgdl)
            scores[doc_id] += boost * qtf * idf_val * num / den

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ÉTAPE 3b : SIMILARITÉ COSINUS
def cosine_similarity(vec1, vec2):
    dot   = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in vec2)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def vectorize_query(query_text, idf):
    tokens = tokenize(query_text)
    if not tokens:
        return {}
    tokens = expand_with_synonyms(tokens)
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    max_f = max(counts.values())
    tf_q = {t: c / max_f for t, c in counts.items()}
    return {t: tf_val * idf.get(t, 0) for t, tf_val in tf_q.items() if t in idf}


# ÉTAPE 3c : ROCCHIO PSEUDO-RELEVANCE FEEDBACK
def rocchio_rerank(query_text, initial_results, tfidf, idf,
                   alpha=ROCCHIO_ALPHA, beta=ROCCHIO_BETA,
                   top_docs=ROCCHIO_TOP_DOCS, top_k=TOP_K):
    q_vec = vectorize_query(query_text, idf)
    if not q_vec or not initial_results:
        return initial_results

    pseudo_rel = [doc_id for doc_id, _ in initial_results[:top_docs]]
    centroid = defaultdict(float)
    for doc_id in pseudo_rel:
        for term, val in tfidf[doc_id].items():
            centroid[term] += val / len(pseudo_rel)

    q_new = defaultdict(float)
    for term, val in q_vec.items():
        q_new[term] += alpha * val
    for term, val in centroid.items():
        q_new[term] += beta * val

    scored = [
        (doc_id, cosine_similarity(doc_vec, dict(q_new)))
        for doc_id, doc_vec in tfidf.items()
    ]
    scored = [(d, s) for d, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# PRODUCTION DU FICHIER RÉSULTATS
def write_results(results, output_file):
    with open(output_file, "w") as f:
        for q_id in sorted(results):
            for doc_id, score in results[q_id]:
                f.write(f"{q_id}\t{doc_id}\t{score:.6f}\n")
    print(f"✓ Résultats écrits dans : {output_file}")

# MAIN
def main():
    print("=" * 55)
    print("  MOTEUR DE RECHERCHE CISI")
    print("  BM25 + Rocchio + spaCy + WordNet + Yake")
    print("=" * 55)

    docs    = parse_collection(COLLECTION_FILE)
    queries = parse_queries(QUERIES_FILE)

    print("\nConstruction des index...")
    inverted, bm25_idf, doc_lengths, avgdl = build_bm25(docs)
    tfidf, idf = build_index(docs)

    print("\nRecherche en cours...")
    results = {}
    for q_id, q_text in sorted(queries.items()):
        initial = bm25_search(q_text, inverted, bm25_idf, doc_lengths, avgdl, top_k=TOP_K)
        ranked  = rocchio_rerank(q_text, initial, tfidf, idf, top_k=TOP_K)
        results[q_id] = ranked

    print(f"✓ Recherche effectuée pour {len(results)} requêtes")

    write_results(results, OUTPUT_FILE)
    print(f'→ Pour évaluer : & "C:\\Program Files\\Git\\usr\\bin\\perl.exe" eval.pl {RELEVANCE_FILE} {OUTPUT_FILE}')


if __name__ == "__main__":
    main()
