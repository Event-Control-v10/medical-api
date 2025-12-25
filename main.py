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

# Modèles à jour (Décembre 2025)
MODEL_TEXT = "llama-3.3-70b-versatile"
MODEL_VISION = "llama-3.2-11b-vision-preview" # Version stable actuelle

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

        if not user_text: return {"error": "No instruction"}

        prompt = f"DataFrame df colonnes: {list(df_orig.columns)}. Instruction: {user_text}. Code Python uniquement (df.at[index, 'col'] = val). Pas de blabla."
        chat = client.chat.completions.create(model=MODEL_TEXT, messages=[{"role": "user", "content": prompt}], temperature=0)
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        exec_scope = {"df": df_mod, "pd": pd}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        START_ROW = 5 
        
        # Injection chirurgicale
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

# --- PARTIE SCANNER CORRIGÉE (MODÈLE 11B) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        
        # Redimensionnement préventif (Gain de vitesse et mémoire)
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((800, 800)) 
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Appel au modèle Vision 11B (le remplaçant du 90B)
        response = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extrait le texte de ce document médical pour Réginalde. Organise les infos par rubriques claires."},
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
        print(f"DEBUG OCR ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur Vision : {str(e)}")

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
