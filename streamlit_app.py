import streamlit as st, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from models.recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

@st.cache_resource
def load_model():
    df = pd.read_csv("data/movies.csv")
    m  = MovieRecommender()
    m.fit(df, verbose=False)
    return m

model = load_model()
titles = model._df["title"].tolist()

title = st.selectbox("Pick a movie", titles)
n     = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    recs = model.recommend(title, n=n)
    st.dataframe(recs, use_container_width=True)