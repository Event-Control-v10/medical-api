import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import openpyxl
from openpyxl import load_workbook
from groq import Groq
import io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/process")
async def process_medical_file(
    file: UploadFile = File(...),
    instruction: str = Form(None),
    audio: UploadFile = File(None)
):
    # 1. Lecture du fichier original
    file_bytes = await file.read()
    
    # On charge le DF pour l'IA (On saute les 3 lignes d'en-tête comme sur sa photo)
    df_original = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
    df_modified = df_original.copy()

    # 2. Gestion de l'instruction (Vocal ou Texte)
    user_text = instruction
    if audio:
        audio_content = await audio.read()
        trans = client.audio.transcriptions.create(
            file=("audio.wav", audio_content),
            model="whisper-large-v3",
            language="fr"
        )
        user_text = trans.text

    if not user_text:
        raise HTTPException(status_code=400, detail="Aucune instruction reçue")

    # 3. Demande de code à l'IA (Plus stricte)
    prompt = f"""
    Tu es un expert Pandas. Modifie le DataFrame 'df' (Colonnes: {list(df_original.columns)}).
    Instruction: "{user_text}"
    RÈGLES:
    - Réponds UNIQUEMENT avec le code Python.
    - Utilise df.at[index, 'nom_colonne'] = valeur.
    - Pas de markdown, pas de blabla.
    """
    
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

    # 4. Exécution du changement sur le DataFrame
    try:
        # On définit 'df' pour l'IA
        local_scope = {"df": df_modified, "pd": pd}
        exec(code, {}, local_scope)
        df_modified = local_scope["df"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur IA : {str(e)}")

    # 5. Injection CHIRURGICALE avec Openpyxl (Garde le Design)
    # On charge le classeur original avec styles
    wb = load_workbook(io.BytesIO(file_bytes))
    ws = wb.active

    # On compare et on écrit uniquement les changements
    # Les données commencent à la ligne 5 dans son fichier
    START_ROW = 5 
    
    for row_idx in range(len(df_original)):
        for col_idx, col_name in enumerate(df_original.columns):
            val_orig = df_original.iat[row_idx, col_idx]
            val_mod = df_modified.iat[row_idx, col_idx]
            
            # Si la valeur a changé, on l'écrit dans la cellule Excel
            if str(val_orig) != str(val_mod):
                # +1 car Excel commence à 1
                ws.cell(row=START_ROW + row_idx, column=col_idx + 1).value = val_mod

    # Sauvegarde finale
    output_path = "resultat_final.xlsx"
    wb.save(output_path)
    
    return FileResponse(output_path, filename="Rapport_Réginalde_MAJ.xlsx")

@app.get("/healthz")
async def healthz(): return {"status": "ok"}
