"""
Step 3: NLP + Recommendation engine
- Cleans storylines (lowercase, remove punctuation/numbers, tokenize,
  remove stopwords, lemmatize)
- Converts text to vectors with TF-IDF
- Finds the top 5 most similar movies using Cosine Similarity
Run:    python recommender.py   (quick test)
"""
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_CSV = "imdb_2024_movies.csv"

# Download NLTK data once (skips if already downloaded)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase -> remove punctuation/numbers -> tokenize -> remove stopwords -> lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    tokens = text.split()                            # tokenize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


class MovieRecommender:
    def __init__(self, csv_path: str = DATA_CSV):
        self.df = pd.read_csv(csv_path).dropna(subset=["Movie Name", "Storyline"])
        self.df = self.df.drop_duplicates(subset="Movie Name").reset_index(drop=True)
        self.df["clean_story"] = self.df["Storyline"].apply(clean_text)

        # TF-IDF: important words get high weight, common words get low weight
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.df["clean_story"])

    def recommend(self, storyline: str, top_n: int = 5) -> pd.DataFrame:
        """Return the top_n movies whose storyline is most similar to the input."""
        query_vec = self.vectorizer.transform([clean_text(storyline)])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_n]

        result = self.df.iloc[top_idx][["Movie Name", "Storyline"]].copy()
        result["Similarity"] = (scores[top_idx] * 100).round(2)   # in %
        return result.reset_index(drop=True)


if __name__ == "__main__":
    rec = MovieRecommender()
    print(f"Loaded {len(rec.df)} movies | TF-IDF vocabulary size: "
          f"{len(rec.vectorizer.vocabulary_)}\n")

    sample = ("A young wizard begins his journey at a magical school where he "
              "makes friends and enemies, facing dark forces along the way.")
    print("Input storyline:", sample, "\n")
    print(rec.recommend(sample).to_string())