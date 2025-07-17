import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib

# Carregar base de vagas (pré-processada com candidatos e vagas)
df = pd.read_csv("base_tratada.csv")

# Carregar modelo SBERT já utilizado no projeto
sbert = joblib.load("models/sbert_encoder.pkl")

# Preparar base de vagas únicas
vagas = df.drop_duplicates(subset=["codigo_vaga"]).copy()
vagas = vagas.dropna(subset=["titulo_vaga", "principais_atividades", "competencias"])

# Unir colunas em um campo de texto único para cada vaga
vagas["descricao_vaga"] = (
    vagas["titulo_vaga"].fillna("") + " " +
    vagas["principais_atividades"].fillna("") + " " +
    vagas["competencias"].fillna("")
)

# Gerar embeddings das vagas uma única vez
print("🔄 Gerando embeddings das vagas...")
vagas_embeddings = sbert.encode(vagas["descricao_vaga"].tolist(), show_progress_bar=True)


def recomendar_vagas(cv_texto, top_k=3):
    """Retorna as top K vagas mais similares ao perfil do candidato"""
    candidato_embedding = sbert.encode([cv_texto])
    similaridades = cosine_similarity(candidato_embedding, vagas_embeddings)[0]
    top_indices = np.argsort(similaridades)[::-1][:top_k]

    resultados = []
    for idx in top_indices:
        vaga = vagas.iloc[idx]
        resultados.append({
            "codigo_vaga": vaga["codigo_vaga"],
            "titulo_vaga": vaga["titulo_vaga"],
            "similaridade": round(similaridades[idx], 3),
            "principais_atividades": vaga["principais_atividades"],
            "competencias": vaga["competencias"]
        })
    return resultados


# Exemplo de uso
if __name__ == "__main__":
    exemplo_cv = df["cv_pt"].dropna().iloc[0]
    recomendacoes = recomendar_vagas(exemplo_cv)
    for i, vaga in enumerate(recomendacoes, 1):
        print(f"\n🏷️ Vaga {i}:")
        print(f"Código: {vaga['codigo_vaga']}")
        print(f"Título: {vaga['titulo_vaga']}")
        print(f"Similaridade: {vaga['similaridade']}")
        print(f"Atividades: {vaga['principais_atividades'][:150]}...")
        print(f"Competências: {vaga['competencias'][:150]}...")
