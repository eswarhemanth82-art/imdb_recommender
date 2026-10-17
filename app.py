"""
Step 4: Streamlit app - IMDb 2024 Movie Recommender (storyline based)
Run:    python -m streamlit run app.py
"""
import matplotlib.pyplot as plt
import streamlit as st

from recommender import MovieRecommender

st.set_page_config(page_title="IMDb Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_model():
    # Loads CSV + builds TF-IDF matrix only once (fast after first load)
    return MovieRecommender("imdb_2024_movies.csv")


rec = load_model()

# ---------- Sidebar ----------
st.sidebar.title("📊 Dataset Info")
st.sidebar.metric("Movies (IMDb 2024)", len(rec.df))
st.sidebar.metric("TF-IDF vocabulary", len(rec.vectorizer.vocabulary_))
top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Tech: Selenium · Pandas · NLTK · TF-IDF · Cosine Similarity · Streamlit")

# ---------- Main ----------
st.title("🎬 IMDb 2024 Movie Recommender")
st.write("Type a movie storyline and get the most similar **2024 movies** "
         "based on their plot, using **TF-IDF + Cosine Similarity**.")

EXAMPLES = {
    "— pick an example —": "",
    "Wizard school": "A young wizard begins his journey at a magical school where he "
                     "makes friends and enemies, facing dark forces along the way.",
    "Revenge thriller": "A retired soldier seeks revenge on the gang that killed his family.",
    "Space survival": "Astronauts stranded on a distant planet fight to survive and return to Earth.",
    "Horror house": "A family moves into an old house and discovers a terrifying evil presence.",
}
choice = st.selectbox("Try an example storyline:", list(EXAMPLES.keys()))
storyline = st.text_area("Enter a movie storyline:", value=EXAMPLES[choice], height=120,
                         placeholder="e.g. A detective hunts a serial killer in a rainy city...")

if st.button("🔍 Recommend Movies", type="primary"):
    if len(storyline.strip().split()) < 3:
        st.warning("Please enter a longer storyline (at least a few words).")
    else:
        results = rec.recommend(storyline, top_n=top_n)

        if results["Similarity"].max() == 0:
            st.error("No similar movies found. Try different words in your storyline.")
        else:
            st.subheader(f"Top {top_n} Recommended Movies")
            left, right = st.columns([3, 2])

            with left:
                for i, row in results.iterrows():
                    with st.container(border=True):
                        st.markdown(f"### {i + 1}. {row['Movie Name']}")
                        st.write(row["Storyline"])
                        st.progress(min(row["Similarity"] / 100, 1.0),
                                    text=f"Similarity: {row['Similarity']}%")

            with right:
                st.markdown("#### 📈 Similarity Scores")
                fig, ax = plt.subplots(figsize=(5, 0.6 * top_n + 1))
                ax.barh(results["Movie Name"][::-1], results["Similarity"][::-1],
                        color="#f5c518")
                ax.set_xlabel("Cosine Similarity (%)")
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)