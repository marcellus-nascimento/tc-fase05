# src/preprocessing.py
import pandas as pd


def load_and_clean_data(path: str) -> pd.DataFrame:
    # Carrega o Excel unificado
    df = pd.read_excel("dados_unificados.xlsx")

    # Cria a variável alvo (1 se foi contratado pela Decision)
    df['foi_contratado'] = df['situacao_candidato'].str.strip().str.lower().apply(
        lambda x: 1 if x == 'contratado pela decision' else 0
    )

    # Renomear colunas para consistência
    df = df.rename(columns={
        'nivel_ingles_x': 'nivel_ingles',
        'nivel_espanhol_x': 'nivel_espanhol'
    })

    # Selecionar colunas relevantes (candidato + vaga + target)
    df = df[[
        'codigo_profissional', 'cv_pt', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol',
        'area_atuacao', 'objetivo_profissional', 'conhecimentos_tecnicos',
        'codigo_vaga', 'titulo_vaga', 'nivel_profissional', 'tipo_contratacao',
        'areas_atuacao', 'principais_atividades', 'competencias',
        'foi_contratado'
    ]]

    # Remover registros com campos essenciais ausentes
    df = df.dropna(subset=['cv_pt', 'titulo_vaga', 'principais_atividades', 'competencias'])

    # Resetar índice
    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    path = "../data/dados_unificados.xlsx"
    df_tratado = load_and_clean_data(path)
    df_tratado.to_parquet("base_tratada.parquet", index=False)
    print("✅ Base tratada salva com sucesso.")