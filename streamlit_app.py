import streamlit as st
import requests

TMDB_API_KEY = "127f1692c84fafe27bf62a1540888e59"
FASTAPI_URL = "FASTAPI_URL = "https://imdb-sentiment-pipeline.onrender.com/predict"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="Análise de Sentimentos", page_icon="🎬", layout="centered")
st.title("🎬 Análise de Sentimentos de Filmes")
st.caption("Busque um filme, escreva uma review em inglês e descubra o sentimento.")

movie_name = st.text_input("Nome do filme", placeholder="Ex: The Dark Knight")

if movie_name:
    resp = requests.get(TMDB_SEARCH_URL, params={"api_key": TMDB_API_KEY, "query": movie_name, "language": "pt-BR"})
    results = resp.json().get("results", [])

    if results:
        movie = results[0]
        col1, col2 = st.columns([1, 2])
        with col1:
            if movie.get("poster_path"):
                st.image(f"{TMDB_IMAGE_URL}{movie['poster_path']}", width=200)
        with col2:
            st.subheader(movie.get("title", ""))
            if movie.get("release_date"):
                st.caption(movie["release_date"][:4])
            st.write(movie.get("overview", ""))
    else:
        st.warning("Filme não encontrado.")

st.divider()

review = st.text_area("Escreva sua review (em inglês)", placeholder="This movie was...", height=150)

if st.button("Analisar Sentimento", type="primary"):
    if not review.strip():
        st.error("Escreva uma review antes de analisar.")
    else:
        with st.spinner("Analisando..."):
            result = requests.post(FASTAPI_URL, json={"text": review})

        if result.status_code == 200:
            data = result.json()
            if data["sentiment"] == "positive":
                st.success(f"✅ Sentimento: **POSITIVO**")
            else:
                st.error(f"❌ Sentimento: **NEGATIVO**")
            st.metric("Confiança", f"{data['confidence'] * 100:.1f}%")
        else:
            st.error("Erro ao conectar com a API. Verifique se ela está rodando.")