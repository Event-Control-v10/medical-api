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
import google.generativeai as genai

app = FastAPI()

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

# CONFIGURATION GEMINI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model_text = genai.GenerativeModel('gemini-1.5-flash')
# On utilise le même modèle pour la vision car 1.5 Flash gère tout

@app.get("/healthz")
async def healthz(): return {"status": "ok"}

# --- EXCEL : MODIFICATION ---
@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None)):
    try:
        file_bytes = await file.read()
        df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
        df_mod = df_orig.copy()

        if not instruction:
            return {"error": "Aucune instruction reçue"}

        # PROMPT POUR GEMINI
        prompt = f"""
        Tu es un expert en Python Pandas. Modifie le DataFrame 'df' (Colonnes: {list(df_orig.columns)}).
        Instruction de Réginalde : "{instruction}"
        
        RÈGLES :
        - Réponds UNIQUEMENT avec le code Python.
        - Utilise df.at[index, 'colonne'] pour être précis.
        - Pas de texte avant ou après, pas de ```python.
        """
        
        response = model_text.generate_content(prompt)
        code = response.text.strip().replace("```python", "").replace("```", "")

        # Exécution
        exec_scope = {"df": df_mod, "pd": pd, "np": np}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        # Sauvegarde chirurgicale (Design)
        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        START_ROW = 5
        
        # Nettoyage et injection
        for row in ws.iter_rows(min_row=START_ROW, max_row=ws.max_row):
            for cell in row: cell.value = None

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

# --- SCANNER : YEUX BIONIQUES GEMINI ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        
        # Amélioration image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
        _, buffer = cv2.imencode('.jpg', enhanced)
        
        # Préparation pour Gemini Vision
        contents = [
            "Extrait le texte de ce document médical pour Réginalde. Organise-le de façon très claire et aérée.",
            {"mime_type": "image/jpeg", "data": buffer.tobytes()}
        ]
        
        response = model_text.generate_content(contents)
        b64_img = base64.b64encode(buffer).decode('utf-8')

        return {
            "text": response.text, 
            "image": f"data:image/jpeg;base64,{b64_img}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
