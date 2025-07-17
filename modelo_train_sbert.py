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


# Carregar base tratada
df = pd.read_csv("base_tratada.csv")

# Manter apenas os dados com texto e target
df = df[["cv_pt", "foi_contratado"]].dropna()

# Pré-processamento
print("🔄 Limpando textos...")
df["cv_pt_limpo"] = df["cv_pt"].apply(preprocess)

# Separar X e y
X = df["cv_pt_limpo"].tolist()
y = df["foi_contratado"].values

# Embeddings com SBERT
print("🔄 Gerando embeddings com SBERT...")
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
sbert = SentenceTransformer(model_name)
X_embeddings = sbert.encode(X, batch_size=32, show_progress_bar=True)


# Validação cruzada com balanceamento simples
def balancear_positivos(X, y):
    df = pd.DataFrame(X)
    df["target"] = y
    min_class = df[df["target"] == 1]
    maj_class = df[df["target"] == 0]
    min_upsampled = min_class.sample(n=len(maj_class), replace=True, random_state=42)
    df_bal = pd.concat([maj_class, min_upsampled])
    y_bal = df_bal["target"].values
    X_bal = df_bal.drop(columns=["target"]).values
    return X_bal, y_bal


X_bal, y_bal = balancear_positivos(X_embeddings, y)

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
    y_pred = modelo.predict(X_embeddings)
    f1 = f1_score(y, y_pred)
    print("Relatório no conjunto completo:")
    print(classification_report(y, y_pred))

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_modelo = modelo
        joblib.dump(melhor_modelo, "models/melhor_modelo_sbert.pkl")
        joblib.dump(sbert, "models/sbert_encoder.pkl")

print(f"\n✅ Melhor modelo salvo: {melhor_modelo.__class__.__name__} (F1 = {melhor_f1:.4f})")
