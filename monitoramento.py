# monitoramento.py
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

# Carregar dados recentes simulados
df_novo = pd.read_csv("dados_recentes.csv")  # precisa criar
df_referencia = pd.read_csv("base_tratada.csv").dropna()

# Comparar apenas colunas comuns
colunas = list(set(df_novo.columns) & set(df_referencia.columns))
df_novo = df_novo[colunas]
df_referencia = df_referencia[colunas]

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_referencia, current_data=df_novo)
report.save_html("drift_report.html")
print("✅ Relatório de drift gerado: drift_report.html")
