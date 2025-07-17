import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# 1. Carregar a base tratada
df = pd.read_csv("base_tratada.csv")

# 2. Selecionar vagas únicas
df_vagas = df[["codigo_vaga", "titulo_vaga", "principais_atividades", "competencias"]].drop_duplicates(subset="codigo_vaga")

# 3. Preencher campos nulos
df_vagas["principais_atividades"] = df_vagas["principais_atividades"].fillna("")
df_vagas["competencias"] = df_vagas["competencias"].fillna("")

# 4. Concatenar para gerar texto completo da vaga
df_vagas["texto_vaga"] = (
    df_vagas["titulo_vaga"] + " " +
    df_vagas["principais_atividades"] + " " +
    df_vagas["competencias"]
)

# 5. Gerar embeddings com SBERT
modelo_sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = modelo_sbert.encode(df_vagas["texto_vaga"].tolist(), show_progress_bar=True)

# 6. Salvar os artefatos
#os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

df_vagas.to_csv("vagas_unicas.csv", index=False)
np.save("models/vagas_embeddings.npy", embeddings)

print("✅ Embeddings gerados e salvos com sucesso!")
