import os
import io
import time
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openpyxl import load_workbook
from groq import Groq
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR): os.makedirs(HISTORY_DIR)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CONFIGURATION DES MODÈLES (VERSION DÉCEMBRE 2025) ---
MODEL_EXCEL = "llama-3.3-70b-versatile"
MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct" # Nouveau modèle Llama 4

@app.get("/healthz")
async def healthz(): return {"status": "ok"}

# --- PARTIE EXCEL ---
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

        if not user_text: return {"error": "Aucune consigne"}

        prompt = f"DataFrame df colonnes: {list(df_orig.columns)}. Instruction: {user_text}. Code Python uniquement (df.at[index, 'colonne'] = val). Pas de blabla."
        chat = client.chat.completions.create(
            model=MODEL_EXCEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0
        )
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        exec_scope = {"df": df_mod, "pd": pd}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        START_ROW = 5 
        
        for r_idx, row in enumerate(df_mod.values):
            for c_idx, val in enumerate(row):
                ws.cell(row=START_ROW + r_idx, column=c_idx + 1).value = val

        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"Reginalde_{ts}.xlsx"
        fpath = os.path.join(HISTORY_DIR, fname)
        wb.save(fpath)
        return FileResponse(fpath, filename=fname)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- PARTIE SCANNER (VERSION LLAMA 4) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img.thumbnail((1024, 1024)) 
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Utilisation du nouveau modèle Llama 4 Multimodal
        response = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tu es un assistant administratif expert en extraction de données (OCR Intelligent).
Ta mission est STRICTEMENT d'extraire les informations visibles sur l'image fournie.

Ne fais aucune analyse médicale. Ne donne pas de conseils. Ne juge pas le contenu.
Ton seul but est de remplir les champs ci-dessous.

Si l'image contient les informations, extrais-les.
Si une information n'est pas présente, écris "Non spécifié".

Renvoie UNIQUEMENT un résultat structuré comme ceci :

1. Nom de l'établissement (Hôpital, Clinique, ou Titre du document) :
2. Nom du patient (ou de la personne concernée) :
3. Numéro de dossier / ID :
4. Date du document :
5. Résumé court du contenu (max 1 phrase) :

Si le document n'est pas médical (ex: un cours, une facture), extrais quand même les titres et les noms visibles.."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
        )
        
        return {
            "text": response.choices[0].message.content,
            "image": f"data:image/jpeg;base64,{base64_image}"
        }
    except Exception as e:
        print(f"ERREUR SCANNER : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur Vision : {str(e)}")

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
