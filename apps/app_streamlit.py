import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import util
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Recrutamento Inteligente", layout="wide")
st.title("📋 Análise de Candidato e Recomendação de Vagas")

# -----------------
# Barra Lateral
# -----------------
with st.sidebar:
    st.markdown("## ℹ️ Sobre o App")
    st.markdown(
        "Este aplicativo analisa o perfil de um candidato a partir do seu CV e dados complementares, "
        "estimando sua chance de contratação e recomendando vagas alinhadas com seu perfil. "
        "Os resultados são baseados em modelos de machine learning e embeddings semânticos."
    )

# -----------------
# Carregamento Seguro
# -----------------
MODELS_PATH = "models"
try:
    modelo = joblib.load(os.path.join(MODELS_PATH, "melhor_modelo_multimodal.pkl"))
    cat_encoder = joblib.load(os.path.join(MODELS_PATH, "cat_encoder.pkl"))
    sbert_encoder = joblib.load(os.path.join(MODELS_PATH, "sbert_encoder.pkl"))
    embeddings_vagas = np.load(os.path.join(MODELS_PATH, "vagas_embeddings.npy"))
    df_vagas = pd.read_csv("vagas_unicas.csv")
except Exception as e:
    st.error(f"Erro ao carregar modelos ou dados: {e}")
    st.stop()

# -----------------
# Funções
# -----------------
def gerar_embedding(texto):
    return sbert_encoder.encode([texto])[0]

def recomendar_vagas(embedding_cv, top_k=3):
    scores = util.cos_sim(embedding_cv, embeddings_vagas)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    vagas = []
    for idx in top_indices:
        row = df_vagas.iloc[idx]
        vagas.append({
            "codigo": int(row["codigo"]),
            "titulo": row["titulo_vaga"],
            "similaridade": round(float(scores[idx]), 4),
            "atividades": row["atividades_principais"],
            "competencias": row["requisitos"]
        })
    return vagas

# -----------------
# Formulário
# -----------------
with st.form("formulario"):
    st.subheader("Preencha os dados do candidato")
    cv_texto = st.text_area("Resumo do CV (campo livre)", height=250)
    col1, col2, col3 = st.columns(3)
    with col1:
        nivel_academico = st.selectbox("Nível Acadêmico", ["Superior Incompleto", "Superior Completo", "Pós", "Mestrado", "Doutorado"])
    with col2:
        nivel_ingles = st.selectbox("Nível de Inglês", ["Básico", "Intermediário", "Avançado", "Fluente"])
    with col3:
        nivel_espanhol = st.selectbox("Nível de Espanhol", ["Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"])

    tipo_contratacao = st.radio("Tipo de Contratação", ["PJ", "CLT"])
    area_atuacao = st.selectbox("Área de Atuação", ["Dados", "Desenvolvimento", "Infraestrutura", "Produto", "Design", "Negócios"])

    submit = st.form_submit_button("Analisar Candidato")

# -----------------
# Resultado
# -----------------
if submit:
    if not cv_texto.strip():
        st.warning("⚠️ Por favor, preencha o campo de CV para prosseguir com a análise.")
    else:
        with st.spinner("🔎 Analisando perfil e buscando vagas ideais..."):
            try:
                df_input = pd.DataFrame([{
                    "cv_pt": cv_texto,
                    "nivel_academico": nivel_academico,
                    "nivel_ingles": nivel_ingles,
                    "nivel_espanhol": nivel_espanhol,
                    "tipo_contratacao": tipo_contratacao,
                    "area_atuacao": area_atuacao
                }])

                embedding_cv = gerar_embedding(cv_texto)
                cat_encoded = cat_encoder.transform(df_input.drop(columns=["cv_pt"]))
                X_input = np.hstack([embedding_cv.reshape(1, -1), cat_encoded.toarray()])
                prob = modelo.predict_proba(X_input)[0, 1]
                recomendacoes = recomendar_vagas(embedding_cv)

                st.success(f"✅ Probabilidade de Contratação: **{round(prob * 100, 2)}%**")

                st.markdown("---")
                st.subheader("🔍 Vagas Recomendadas")
                for vaga in recomendacoes:
                    st.markdown(f"### {vaga['titulo']} (Código: {vaga['codigo']})")
                    st.markdown(f"**Similaridade:** {vaga['similaridade']}")
                    st.markdown(f"**Atividades:**\n{vaga['atividades']}")
                    st.markdown(f"**Competências:**\n{vaga['competencias']}")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Ocorreu um erro durante a análise: {e}")

