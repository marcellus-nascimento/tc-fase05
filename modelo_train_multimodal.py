import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Baixar recursos do nltk
nltk.download('stopwords')
nltk.download('rslp')
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

# Função de pré-processamento
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Carregar base
df = pd.read_csv("base_tratada.csv")
df = df.dropna(subset=["cv_pt", "foi_contratado"])

# Pré-processamento do texto
print("🔄 Limpando textos...")
df["cv_pt_limpo"] = df["cv_pt"].apply(preprocess)

# Gerar embeddings com SBERT
print("🔄 Gerando embeddings SBERT...")
sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = sbert.encode(df["cv_pt_limpo"].tolist(), batch_size=32, show_progress_bar=True)

# Variáveis categóricas
cat_features = [
    "nivel_academico", "tipo_contratacao", "area_atuacao",
    "nivel_ingles", "nivel_espanhol"
]
df = df.dropna(subset=cat_features)
df = df.reset_index(drop=True)

# Ajustar embeddings ao mesmo tamanho do df final (após dropna)
embeddings = embeddings[:df.shape[0]]
y = df["foi_contratado"].values
X_cat = df[cat_features]

# One-hot encoding
print("🧠 Aplicando one-hot nas variáveis...")
onehot = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_features)
])
X_cat_encoded = onehot.fit_transform(X_cat).toarray()

# Concatenar embeddings + categóricas
X_final = np.hstack([embeddings, X_cat_encoded])

# Balanceamento simples
def balancear(X, y):
    df_x = pd.DataFrame(X)
    df_x["target"] = y
    maj = df_x[df_x["target"] == 0]
    min = df_x[df_x["target"] == 1]
    min_upsample = min.sample(n=len(maj), replace=True, random_state=42)
    df_bal = pd.concat([maj, min_upsample])
    y_bal = df_bal["target"].values
    X_bal = df_bal.drop(columns=["target"]).values
    return X_bal, y_bal

X_bal, y_bal = balancear(X_final, y)

# Modelos
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=150, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=150, eval_metric='logloss'),
    "VotingClassifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced')),
        ('xgb', XGBClassifier(n_estimators=100, eval_metric='logloss'))
    ], voting='soft')
}

# Validação cruzada e treino final
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
melhor_f1 = 0
melhor_modelo = None

for nome, modelo in models.items():
    print(f"\n📊 Avaliando {nome}...")
    f1_scores = cross_val_score(modelo, X_bal, y_bal, scoring='f1', cv=cv, n_jobs=-1)
    print(f"F1-score médio (validação cruzada): {f1_scores.mean():.4f}")

    modelo.fit(X_bal, y_bal)
    y_pred = modelo.predict(X_final)
    f1 = f1_score(y, y_pred)
    print("Relatório no conjunto completo:")
    print(classification_report(y, y_pred))

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_modelo = modelo
        joblib.dump(melhor_modelo, "models/melhor_modelo_multimodal.pkl")
        joblib.dump(sbert, "models/sbert_encoder.pkl")
        joblib.dump(onehot, "models/cat_encoder.pkl")

print(f"\n✅ Melhor modelo salvo: {melhor_modelo.__class__.__name__} (F1 = {melhor_f1:.4f})")
