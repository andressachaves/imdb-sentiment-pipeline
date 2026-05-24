# IMDB Sentiment Pipeline

Pipeline completo de análise de sentimentos aplicado ao dataset Stanford IMDB, com modelo em produção, API REST e dashboard interativo.

**[Demo ao vivo](https://imdb-sentiment-streamlit.onrender.com)** | **[API](https://imdb-sentiment-pipeline.onrender.com/docs)** | **[Modelo no Hugging Face](https://huggingface.co/anddz/imdb-sentiment-linearsvc)**

---

## Resultado do modelo

| Métrica | Valor |
|---|---|
| Acurácia | 92,33% |
| F1-macro | 92,33% |
| Amostras de teste | 10.000 |
| Modelo | LinearSVC + TF-IDF |

![Matriz de Confusão](grafico_matriz_confusao.png)

---

## Arquitetura
Etapa 01 — Dataset
Hugging Face Dataset (stanfordnlp/imdb)
│
▼
Etapa 02 — Coleta e ingestão
coleta_dados.ipynb
│
▼
Etapa 03 — Limpeza e processamento
processamento.ipynb
│
▼
Etapa 04 — Modelagem
script_svm.ipynb → LinearSVC + TF-IDF → 92,33% acurácia
│
▼
Hugging Face Hub (modelo hospedado)
│
▼
FastAPI (API REST em produção)
│
▼
Streamlit (dashboard com busca de filmes via TMDB)

---
## Stack
- Python 3
- scikit-learn (TfidfVectorizer + LinearSVC)
- FastAPI + Uvicorn
- Streamlit
- Hugging Face Hub
- TMDB API
- Render (deploy)
- Google Colab + Google Drive (treinamento)
---
## Como executar localmente
**Pré-requisitos:** Python 3.11+
```bash
git clone https://github.com/andressachaves/imdb-sentiment-pipeline.git
cd imdb-sentiment-pipeline
pip install -r requirements.txt
Rodar a API:

python -m uvicorn app.main:app --reload
Rodar o dashboard:

python -m streamlit run streamlit_app.py
Na primeira execução a API baixa o modelo do Hugging Face automaticamente.

API
Endpoint: POST /predict

curl -X POST "https://imdb-sentiment-pipeline.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely amazing!"}'
Resposta:

{
  "sentiment": "positive",
  "confidence": 0.9712,
  "label": 1
}
Limitações conhecidas
Modelo treinado para reviews em inglês — textos em outros idiomas terão baixa confiança
Textos neutros ficam próximos de 50% de confiança — o modelo só conhece positivo e negativo
Negações complexas podem ser mal interpretadas — limitação do TF-IDF que analisa palavras isoladas
Próximos passos
não concluído
Comparação de modelos (Naive Bayes, Logistic Regression, BERT)
não concluído
Suporte a português com modelo multilingual
não concluído
Histórico de análises com gráfico de sentimentos
não concluído
Monitoramento de drift do modelo


