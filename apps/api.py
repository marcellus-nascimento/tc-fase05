from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# Inicializa API
app = FastAPI(title="API Recrutamento - Contratação e Recomendacão de Vaga")

# Define modelo de entrada
class Candidato(BaseModel):
    cv_pt: str
    nivel_academico: str
    nivel_ingles: str
    nivel_espanhol: str
    tipo_contratacao: str
    area_atuacao: str

# Caminho absoluto do diretório raiz (um nível acima de /apps)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Caminhos absolutos
MODELS_PATH = os.path.join(ROOT_DIR, "models")
DATA_PATH = ROOT_DIR  # onde está o vagas_unicas.csv

# Carregando arquivos
modelo_multimodal = joblib.load(os.path.join(MODELS_PATH, "melhor_modelo_multimodal.pkl"))
cat_encoder = joblib.load(os.path.join(MODELS_PATH, "cat_encoder.pkl"))
sbert_encoder = joblib.load(os.path.join(MODELS_PATH, "sbert_encoder.pkl"))
df_vagas = pd.read_csv(os.path.join(DATA_PATH, "vagas_unicas.csv")).fillna("")
embeddings_vagas = np.load(os.path.join(MODELS_PATH, "vagas_embeddings.npy"))

# ------------------
# Funções auxiliares
# ------------------
def gerar_embedding_cv(texto: str):
    return sbert_encoder.encode([texto])[0]

def recomendar_vagas(embedding_cv, top_k=3):
    scores = util.cos_sim(embedding_cv, embeddings_vagas)[0].cpu().numpy()
    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    vagas_recomendadas = []
    for idx in top_indices:
        row = df_vagas.iloc[idx]
        vagas_recomendadas.append({
            "codigo": int(row["codigo_vaga"]),
            "titulo": row["titulo_vaga"],
            "similaridade": round(float(scores[idx]), 4),
            "atividades": row["principais_atividades"],
            "competencias": row["competencias"],
        })
    return vagas_recomendadas

# ------------------
# Rota principal
# ------------------
@app.post("/analisar_candidato")
def analisar_candidato(dados: Candidato):
    # Organiza dados em DataFrame
    df_input = pd.DataFrame([dados.dict()])

    # Gera embedding do CV
    embedding_cv = gerar_embedding_cv(dados.cv_pt)

    # Codifica variáveis categóricas
    cat_features = df_input.drop(columns=["cv_pt"])
    cat_encoded = cat_encoder.transform(cat_features)

    # Junta features
    X_input = np.hstack([embedding_cv.reshape(1, -1), cat_encoded.toarray()])

    # Predição
    prob = modelo_multimodal.predict_proba(X_input)[0, 1]

    # Recomendação
    recomendacoes = recomendar_vagas(embedding_cv)

    return {
        "probabilidade_contratacao": round(float(prob), 4),
        "vagas_recomendadas": recomendacoes
    }
