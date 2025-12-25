import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import openpyxl
from groq import Groq
import io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Remplace par ta clé ou utilise une variable d'environnement
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/process")
async def process(file: UploadFile = File(...), instruction: str = Form(None), audio: UploadFile = File(None)):
    file_bytes = await file.read()
    
    # 1. Transcription si audio présent
    final_text = instruction
    if audio:
        audio_data = await audio.read()
        trans = client.audio.transcriptions.create(file=("a.wav", audio_data), model="whisper-large-v3", language="fr")
        final_text = trans.text

    # 2. Analyse du fichier pour l'IA
    df = pd.read_excel(io.BytesIO(file_bytes), skiprows=3) # On saute les 3 lignes de titre

    # 3. Demande de code à l'IA
    prompt = f"Colonnes: {list(df.columns)}. Données: {df.head(2).to_string()}. Instruction: {final_text}. Écris uniquement le code Python pour modifier 'df'. Pas de texte, pas de balises."
    res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}])
    code = res.choices[0].message.content.strip().replace("```python","").replace("```","")

    # 4. Exécution
    exec(code)

    # 5. Injection dans l'original (Garde le Design)
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes))
    ws = wb.active
    for r_idx, row in df.iterrows():
        for c_idx, value in enumerate(row):
            ws.cell(row=5 + r_idx, column=c_idx + 1).value = value

    out_path = "resultat.xlsx"
    wb.save(out_path)
    return FileResponse(out_path, filename="Rapport_Medical.xlsx")

@app.get("/healthz")
async def healthz(): return {"status": "ok"}
