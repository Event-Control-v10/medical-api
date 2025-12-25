import os
import io
import time
import base64
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openpyxl import load_workbook
from groq import Groq

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

@app.get("/healthz")
async def healthz(): return {"status": "ok"}

# --- EXCEL ---
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

        prompt = f"DataFrame df: {list(df_orig.columns)}. Instruction: {user_text}. Code Python uniquement pour modifier df via df.at."
        chat = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0)
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        exec_scope = {"df": df_mod, "pd": pd, "np": np}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        for r_idx, row in enumerate(df_mod.values):
            for c_idx, val in enumerate(row):
                ws.cell(row=5+r_idx, column=c_idx+1).value = val

        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"Reginalde_{ts}.xlsx"
        fpath = os.path.join(HISTORY_DIR, fname)
        wb.save(fpath)
        return FileResponse(fpath, filename=fname)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SCANNER (CORRIGÉ) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        print("Début du scan...")
        img_bytes = await image.read()
        
        # Encodage direct en Base64 pour l'IA
        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        # Tentative avec le modèle Vision 90B
        # Si ce modèle échoue, Groq renverra une erreur claire dans les logs
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extrait tout le texte de ce document médical. Organise-le bien."},
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
        print(f"Erreur OCR détaillée : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur IA : {str(e)}")
