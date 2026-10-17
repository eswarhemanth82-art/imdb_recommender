# 🎬 IMDb 2024 Movie Recommendation System (Storyline-Based)

A content-based movie recommender that suggests the **top 5 most similar 2024 movies** for any storyline you type, using **NLP (TF-IDF)** and **Cosine Similarity**, with an interactive **Streamlit** app.

**Domain:** Entertainment / Data Analytics / Recommender Systems

---

## 🚀 Features
- **Web scraping with Selenium**: collects ~1000 IMDb 2024 movies (Movie Name + Storyline)
- **NLP preprocessing**: lowercasing, punctuation/number removal, tokenization, stopword removal, lemmatization (NLTK)
- **TF-IDF vectorization** with unigrams + bigrams
- **Cosine Similarity** ranking to find the most similar storylines
- **Streamlit UI**: example storylines, adjustable number of results, movie cards with similarity %, and a Matplotlib similarity bar chart

## 🛠️ Tech Stack
| Area | Tools |
|---|---|
| Language | Python |
| Web Scraping | Selenium |
| Data Handling | Pandas |
| NLP | NLTK, Scikit-learn (TfidfVectorizer) |
| Recommendation | Cosine Similarity |
| UI & Visualization | Streamlit, Matplotlib |

## 📁 Project Structure
```
imdb_recommender/
├── scraper.py             # Selenium scraper -> imdb_2024_movies.csv
├── recommender.py         # Text cleaning + TF-IDF + Cosine Similarity
├── app.py                 # Streamlit web app
├── imdb_2024_movies.csv   # Scraped dataset (Movie Name, Storyline)
├── requirements.txt
└── README.md
```

## ⚙️ How It Works
1. **Scraping**: `scraper.py` opens the IMDb 2024 feature-film search page in Chrome, clicks **"50 more"** repeatedly until ~1000 movies are loaded, extracts each title and plot, removes empty/duplicate rows and saves `imdb_2024_movies.csv`.
2. **Preprocessing**: each storyline is cleaned into meaningful lemmatized tokens (e.g. *"A young wizard begins his journey"* → `young wizard begin journey`).
3. **Vectorization**: TF-IDF turns every storyline into a numeric vector. Rare, meaningful words (e.g. *wizard*, *haunted*) get higher weight than common ones.
4. **Similarity**: the user's storyline is cleaned and vectorized the same way, then compared with every movie using cosine similarity. The highest scores are returned as recommendations.

## ▶️ Run Locally
```bash
git clone https://github.com/<your-username>/imdb_recommender.git
cd imdb_recommender
pip install -r requirements.txt

# (optional) re-scrape fresh data - needs Google Chrome
python scraper.py

# quick test in terminal
python recommender.py

# launch the app
python -m streamlit run app.py
```
Open http://localhost:8501 in your browser.

## 💡 Example
**Input:** *"A family moves into an old house and discovers a terrifying evil presence."*
**Output:** top 5 2024 horror/supernatural movies with the closest storylines, each with its similarity score.

## 📊 Dataset
- **Source:** IMDb, feature films released 1 Jan 2024 to 31 Dec 2024
- **Rows:** 996 movies (after removing empty and duplicate storylines)
- **Columns:** `Movie Name`, `Storyline`

## 🔮 Future Improvements
- Use sentence embeddings (e.g. Sentence-BERT) for meaning-based similarity instead of word overlap
- Add genre, rating and poster data to the recommendations
- Deploy the app on Streamlit Community Cloud

## 👤 Author
**T.S. Hemanth (Hemi)**: AI/ML Project, HCL GUVI