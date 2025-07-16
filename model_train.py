# model_train.py
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sentence_transformers import SentenceTransformer

# 1. Garantir pasta de modelos
os.makedirs("models", exist_ok=True)

# 2. Carregar dados processados
X_train_tfidf = joblib.load("models/X_train.pkl")
X_test_tfidf = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

# 3. Carregar textos crus para embeddings
df = pd.read_csv("base_tratada.csv")
df = df.dropna(subset=["cv_pt", "foi_contratado"])
X_text = df["cv_pt"]
y = df["foi_contratado"]

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Gerar embeddings com SBERT
print("🔄 Gerando embeddings com SBERT...")
model_sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X_train_sbert = model_sbert.encode(X_train_text.tolist(), show_progress_bar=True)
X_test_sbert = model_sbert.encode(X_test_text.tolist(), show_progress_bar=True)

# 5. Função de treino e avaliação
results = []

def train_and_evaluate(name, model, Xtr, Xte, ytr, yte):
    print(f"\n📊 Treinando {name}...")
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    f1 = f1_score(yte, preds)
    print(classification_report(yte, preds))
    results.append({
        "model_name": name,
        "f1_score": f1,
        "model": model
    })

# 6. Treinar modelos
train_and_evaluate("LogReg_TFIDF", LogisticRegression(max_iter=1000), X_train_tfidf, X_test_tfidf, y_train, y_test)
train_and_evaluate("RF_TFIDF", RandomForestClassifier(), X_train_tfidf, X_test_tfidf, y_train, y_test)
train_and_evaluate("LogReg_SBERT", LogisticRegression(max_iter=1000), X_train_sbert, X_test_sbert, y_train_text, y_test_text)
train_and_evaluate("RF_SBERT", RandomForestClassifier(), X_train_sbert, X_test_sbert, y_train_text, y_test_text)

# 7. Selecionar e salvar o melhor modelo
best = max(results, key=lambda x: x["f1_score"])
joblib.dump(best["model"], "models/best_model.pkl")
with open("models/best_model_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Melhor modelo: {best['model_name']}\nF1-score: {best['f1_score']:.4f}")

print(f"\n✅ Melhor modelo: {best['model_name']} com F1-score = {best['f1_score']:.4f}")
