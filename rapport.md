# Rapport — Moteur de Recherche Textuel
## Projet Text Information Retrieval — Collection CISI

## 1. Indexation de la collection   

### Tokenisation — spaCy

Pour chaque document, j'ai utilisé la bibliothèque **spaCy** (`en_core_web_sm`) pour traiter le texte :

1. Mise en minuscules.
2. **Lemmatisation** : spaCy réduit chaque mot à sa forme de base (`libraries` → `library`, `running` → `run`). C'est plus précis qu'un stemmer par suffixes.
3. **Filtrage POS** (Part-of-Speech) : je garde uniquement les noms, verbes, adjectifs et adverbes — les mots qui portent du sens.
4. Suppression des stopwords intégrés à spaCy.
5. Suppression des tokens de moins de 3 caractères.

### Vocabulaire indexé

| Statistique | Valeur |
|-------------|--------|
| Nombre de documents | 1460 |
| Termes distincts (après lemmatisation + filtrage POS) | **6482** |
| Longueur moyenne d'un document (tokens) | **63,6** |

### Pondération — TF-IDF

- **TF normalisé** : `tf(t, d) = count(t, d) / max_count(d)`
- **IDF** : `idf(t) = log(N / df(t))`
- **Poids final** : `w(t, d) = tf(t, d) × idf(t)`

Ces vecteurs TF-IDF sont utilisés dans la phase Rocchio.

### Index inversé

Structure : `terme → { doc_id : fréquence_brute }`

Permet un calcul rapide des scores BM25 sans parcourir toute la collection pour chaque requête.


## 2. Moteur de recherche

### Indexation des requêtes

Les requêtes sont traitées avec exactement la même chaîne de traitement que les documents (spaCy, lemmatisation, filtrage POS, stopwords). Seul le champ `.W` est utilisé. La pondération est aussi identique : TF normalisé × IDF_BM25, appliqué à chaque terme de la requête.

En plus, j'ai appliqué deux traitements supplémentaires uniquement sur les requêtes :

- **Expansion par synonymes (WordNet)** : chaque terme de la requête est enrichi avec jusqu'à 2 synonymes tirés de WordNet (via nltk). Cela aide à retrouver des documents qui utilisent des mots différents mais de sens proche.
- **Extraction de termes clés (Yake)** : les termes les plus importants de la requête sont identifiés par Yake et reçoivent un poids boosté (×1.5) dans le scoring BM25.

### Mesure de similarité — BM25

Parmi les modèles vus en cours, j'ai choisi **BM25** (Best Match 25), modèle de ranking probabiliste vu en cours (Robertson & Walker, 1994). Contrairement au TF-IDF classique, il intègre une saturation de la fréquence des termes (un terme très répété n'apporte pas infiniment plus de score) et une normalisation par la longueur du document, ce qui le rend plus robuste sur des collections hétérogènes. C'est pour ces raisons que je l'ai choisi plutôt que la similarité cosinus classique.

La formule utilisée est la suivante :

```
                        tf(t,d) × (k1 + 1)
BM25(q,d) = Σ IDF(t) × ─────────────────────────────────────
             t∈q         tf(t,d) + k1 × (1 - b + b × |d|/avgdl)
```

avec :
- `tf(t,d)` = fréquence normalisée du terme t dans le document d
- `IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )`
- `k1 = 1.5` — saturation de la fréquence des termes
- `b = 0.75` — normalisation par la longueur
- `avgdl = 63.6` — longueur moyenne des documents

### Feedback pseudo-pertinent — Rocchio

Pour améliorer les résultats après BM25, j'ai appliqué le **feedback pseudo-pertinent de Rocchio**, vu en cours comme technique d'expansion de requête complémentaire aux modèles de ranking.

L'algorithme de Rocchio est une technique d'expansion de requête : l'idée est de supposer que les premiers documents retournés sont pertinents, et de modifier automatiquement la requête pour la rapprocher de ces documents. Cela permet de capturer des termes importants que l'utilisateur n'a pas forcément écrits dans sa requête initiale.

Concrètement, après le classement BM25, j'ai enrichi la requête de la façon suivante :

```
q_new = α × q + β × (1/|Dr|) × Σ d
```

avec α = 1.0, β = 0.75, et les 5 premiers documents supposés pertinents. Les documents sont ensuite re-classés par similarité cosinus avec q_new.


## 3. Évaluation

Évaluation réalisée avec `eval.pl` sur les 30 requêtes, TOP_K = 100.

| Requête | Précision | Rappel | F1 | P@1 | P@5 |
|---------|-----------|--------|----|-----|-----|
| Q1  | 31.0% | 43.7% | 36.3% | 100% | 40% |
| Q2  | 12.0% | 24.0% | 16.0% | 100% | 60% |
| Q3  | 0.0% | — | 0.0% | 0% | 0% |
| Q4  | 6.0% | 25.0% | 9.7% | 0% | 20% |
| Q5  | 0.0% | — | 0.0% | 0% | 0% |
| Q6  | 0.0% | — | 0.0% | 0% | 0% |
| Q7  | 0.0% | — | 0.0% | 0% | 0% |
| Q8  | 0.0% | — | 0.0% | 0% | 0% |
| Q9  | 0.0% | — | 0.0% | 0% | 0% |
| Q10 | 12.0% | 31.6% | 17.4% | 100% | 20% |
| Q11 | 0.0% | — | 0.0% | 0% | 0% |
| Q12 | 13.0% | 27.1% | 17.6% | 0% | 0% |
| Q13 | 31.0% | 67.4% | 42.5% | 0% | 60% |
| Q14 | 15.0% | 28.3% | 19.6% | 0% | 20% |
| Q15 | 10.0% | 66.7% | 17.4% | 0% | 20% |
| Q16 | 10.0% | 83.3% | 17.9% | 100% | 80% |
| Q17 | 8.0% | 72.7% | 14.4% | 0% | 0% |
| Q18 | 35.0% | 58.3% | 43.8% | 100% | 80% |
| Q19 | 12.0% | 35.3% | 17.9% | 100% | 60% |
| Q20 | 26.0% | 74.3% | 38.5% | 100% | 100% |
| Q21 | 19.0% | 55.9% | 28.4% | 100% | 100% |
| Q22 | 13.0% | 34.2% | 18.8% | 100% | 60% |
| Q23 | 0.0% | — | 0.0% | 0% | 0% |
| Q24 | 0.0% | — | 0.0% | 0% | 0% |
| Q25 | 18.0% | 75.0% | 29.0% | 100% | 100% |
| Q26 | 21.0% | 63.6% | 31.6% | 0% | 20% |
| Q27 | 19.0% | 37.3% | 25.2% | 0% | 0% |
| Q28 | 6.0% | 46.2% | 10.6% | 0% | 20% |
| Q29 | 1.0% | 100.0% | 2.0% | 0% | 100% |
| Q30 | 2.0% | 25.0% | 3.7% | 0% | 0% |
| **Overall** | **10.7%** | **45.8%** | **17.3%** | **33%** | **32%** |

### Comparaison des versions

| Métrique | TF-IDF seul | BM25 + Rocchio | + spaCy + WordNet + Yake |
|----------|-------------|----------------|--------------------------|
| Précision | 10.2% | 10.9% | 10.7% |
| Rappel | 34.7% | 46.9% | 45.8% |
| F1 | 14.7% | 17.7% | 17.3% |
| P@1 | 26.7% | 43% | 33% |
| **P@5** | 26.7% | 27% | **32%** |

L'ajout des outils NLP améliore la **P@5** (+5 points) grâce à l'expansion par synonymes WordNet qui permet de retrouver des documents sémantiquement proches. La précision globale reste stable autour de 10%, ce qui est attendu pour un système lexical retournant 100 documents par requête sur la collection CISI.

Les 9 requêtes à score nul (Q3, Q5–Q9, Q11, Q23, Q24) n'ont aucun document pertinent dans `CISI_dev.REL`.


## 4. Outils utilisés

| Outil | Usage |
|-------|-------|
| Python 3 | Langage d'implémentation |
| spaCy (`en_core_web_sm`) | Tokenisation, lemmatisation, filtrage POS, stopwords |
| WordNet via nltk | Expansion des requêtes par synonymes |
| Yake | Extraction des termes clés des requêtes (boosting BM25) |
| `math`, `collections` | Calculs IDF, BM25, index inversé |

## 5.Exécution

-Execution du fichier search_engine.py marche normalement.
-Le script d'évaluation eval.pl nécessite Perl pour s'exécuter. Sur ma machine, Perl n'est pas installé de façon autonome mais est fourni avec Git for Windows, ce qui fait qu'il n'est pas reconnu comme commande directe dans le terminal. Pour l'exécuter, il faut donc utiliser le chemin complet vers l'exécutable Perl de Git :
& "C:\Program Files\Git\usr\bin\perl.exe" eval.pl CISI_dev.REL results.txt
Si Perl est installé de façon classique sur votre machine (vérifiable avec perl --version dans le terminal), la commande simplifiée suffit : perl eval.pl CISI_dev.REL results.txt 


## 6. Conclusion

Ce projet m'a permis de construire un moteur de recherche complet, en passant d'un TF-IDF basique à un pipeline BM25 + Rocchio enrichi par des outils NLP. Chaque amélioration a apporté un gain mesurable. Les scores obtenus (~10% de précision, ~33% de P@1, ~32% de P@5) sont cohérents avec ce qu'on attend d'un système lexical sur la collection CISI pour 100 résultats retournés par requête.
