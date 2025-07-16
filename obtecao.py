import json
import pandas as pd
import re

# ======== 1. Carregar os arquivos JSON ========
with open('applicants.json', 'r', encoding='utf-8') as f:
    applicants_raw = json.load(f)

with open('prospects.json', 'r', encoding='utf-8') as f:
    prospects_raw = json.load(f)

with open('jobs.json', 'r', encoding='utf-8') as f:
    jobs_raw = json.load(f)

# ======== 2. Processar Applicants.json ========
applicants_data = []
for applicant_id, info in applicants_raw.items():
    applicants_data.append({
        'codigo_profissional': applicant_id,
        'nome_applicant': info['infos_basicas'].get('nome'),
        'email': info['infos_basicas'].get('email'),
        'telefone': info['infos_basicas'].get('telefone'),
        'objetivo_profissional': info['infos_basicas'].get('objetivo_profissional'),
        'nivel_academico': info['formacao_e_idiomas'].get('nivel_academico'),
        'nivel_ingles': info['formacao_e_idiomas'].get('nivel_ingles'),
        'nivel_espanhol': info['formacao_e_idiomas'].get('nivel_espanhol'),
        'area_atuacao': info['informacoes_profissionais'].get('area_atuacao'),
        'conhecimentos_tecnicos': info['informacoes_profissionais'].get('conhecimentos_tecnicos'),
        'cv_pt': info.get('cv_pt', '')
    })

df_applicants = pd.DataFrame(applicants_data)

# ======== 3. Processar Prospects.json ========
prospects_data = []
for job_id, job_info in prospects_raw.items():
    for prospect in job_info.get("prospects", []):
        prospects_data.append({
            'codigo_vaga': job_id,
            'titulo_vaga_prospect': job_info.get('titulo', ''),
            'modalidade': job_info.get('modalidade', ''),
            'codigo_profissional': prospect.get('codigo'),
            'nome_candidato': prospect.get('nome'),
            'situacao_candidato': prospect.get('situacao_candidado'),
            'data_candidatura': prospect.get('data_candidatura'),
            'ultima_atualizacao': prospect.get('ultima_atualizacao'),
            'comentario': prospect.get('comentario'),
            'recrutador': prospect.get('recrutador')
        })

df_prospects = pd.DataFrame(prospects_data)

# ======== 4. Processar Jobs.json ========
jobs_data = []
for job_id, job_info in jobs_raw.items():
    info = job_info.get('informacoes_basicas', {})
    perfil = job_info.get('perfil_vaga', {})
    jobs_data.append({
        'codigo_vaga': job_id,
        'titulo_vaga': info.get('titulo_vaga'),
        'vaga_sap': info.get('vaga_sap'),
        'cliente': info.get('cliente'),
        'tipo_contratacao': info.get('tipo_contratacao'),
        'local_trabalho': perfil.get('local_trabalho'),
        'nivel_profissional': perfil.get('nivel profissional'),
        'nivel_ingles': perfil.get('nivel_ingles'),
        'nivel_espanhol': perfil.get('nivel_espanhol'),
        'areas_atuacao': perfil.get('areas_atuacao'),
        'principais_atividades': perfil.get('principais_atividades'),
        'competencias': perfil.get('competencia_tecnicas_e_comportamentais')
    })

df_jobs = pd.DataFrame(jobs_data)

# ======== 5. Juntar tudo ========
df_merge1 = pd.merge(df_prospects, df_applicants, on='codigo_profissional', how='left')
df_final = pd.merge(df_merge1, df_jobs, on='codigo_vaga', how='left')

def clean_illegal_chars(val):
    if isinstance(val, str):
        # Remove caracteres de controle não permitidos no Excel (exceto \n e \t)
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', val)
    return val

# Aplicar a função em todo o DataFrame
df_final_clean = df_final.applymap(clean_illegal_chars)

# Agora pode exportar com segurança
df_final_clean.to_excel("dados_unificados.xlsx", index=False)