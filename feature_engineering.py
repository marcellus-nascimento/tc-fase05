import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import joblib
import os

# 1. Carregamento da base
df = pd.read_csv("base_tratada.csv")

# 2. Seleção de colunas
coluna_texto = "cv_pt"
colunas_categoricas = [
    "nivel_academico", "nivel_ingles", "nivel_espanhol",
    "tipo_contratacao", "area_atuacao"
]
target = "foi_contratado"

# 3. Drop de linhas com texto nulo
df = df.dropna(subset=[coluna_texto, target])

# 4. Divisão em X e y
X = df[[coluna_texto] + colunas_categoricas]
y = df[target]

# 5. Pipeline de transformação
preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=1000), "cv_pt"),
        ("onehot", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), colunas_categoricas)
    ]
)

# 6. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Balanceamento simples (up-sampling dos positivos)
X_train["foi_contratado"] = y_train
df_majority = X_train[X_train["foi_contratado"] == 0]
df_minority = X_train[X_train["foi_contratado"] == 1]
df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)
X_train_balanced = pd.concat([df_majority, df_minority_upsampled])
y_train_balanced = X_train_balanced["foi_contratado"]
X_train_balanced = X_train_balanced.drop(columns=["foi_contratado"])

# 8. Fit transformador
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
X_train_transformed = pipeline.fit_transform(X_train_balanced)
X_test_transformed = pipeline.transform(X_test)

# 9. Salvar os arquivos processados
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/transformer.pkl")
joblib.dump(X_train_transformed, "models/X_train.pkl")
joblib.dump(y_train_balanced, "models/y_train.pkl")
joblib.dump(X_test_transformed, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("Feature engineering finalizado e arquivos salvos.")

