import os
import io
import time
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openpyxl import load_workbook
from groq import Groq

app = FastAPI()

# CORS ultra-large pour autoriser les requêtes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/healthz")
async def healthz(): return {"status": "ok"}

# --- EXCEL : COMPRÉHENSION BOOSTÉE ---
@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None), audio: UploadFile = File(None)):
    try:
        file_bytes = await file.read()
        df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
        df_mod = df_orig.copy()

        user_text = instruction
        if audio:
            audio_data = await audio.read()
            trans = client.audio.transcriptions.create(file=("a.wav", audio_data), model="whisper-large-v3", language="fr")
            user_text = trans.text

        # Système de prompt renforcé pour une meilleure compréhension
        system_prompt = "Tu es un expert en manipulation de données médicales pour Réginalde. Ton but est de générer du code Python Pandas précis."
        prompt = f"""
        DataFrame actuel (df) :
        Colonnes : {list(df_orig.columns)}
        Aperçu : {df_orig.head(5).to_string()}

        DEMANDE DE RÉGINALDE : "{user_text}"

        RÈGLES :
        1. Modifie l'objet 'df' directement (ex: df.at[index, 'colonne'] = valeur).
        2. Si la demande concerne un nom de docteur, cherche la ligne correspondante dans la colonne 'Nom du Docteur/ Nom du Prospect'.
        3. Réponds UNIQUEMENT avec du code Python. Pas de texte, pas de markdown.
        """
        
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0
        )
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        exec_scope = {"df": df_mod, "pd": pd, "np": np}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        # Sauvegarde chirurgicale (Garde le Design)
        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        START_ROW = 5
        
        for r_idx, row_values in enumerate(df_mod.values):
            for c_idx, value in enumerate(row_values):
                ws.cell(row=START_ROW + r_idx, column=c_idx + 1).value = value

        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"Reginalde_{ts}.xlsx"
        fpath = os.path.join(HISTORY_DIR, fname)
        wb.save(fpath)
        return FileResponse(fpath, filename=fname)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SCANNER : MODÈLE RÉCENT ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        enhanced = cv2.convertScaleAbs(img, alpha=1.4, beta=10)
        _, buffer = cv2.imencode('.jpg', enhanced)
        b64_img = base64.b64encode(buffer).decode('utf-8')

        res = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview", # Modèle à jour
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Analyse ce document pour Réginalde. Extrait tout le texte lisible de façon très organisée."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}]
        )
        return {"text": res.choices[0].message.content, "image": f"data:image/jpeg;base64,{b64_img}"}
    except Exception as e:
        return {"text": "Erreur lors de l'analyse IA", "error": str(e)}
