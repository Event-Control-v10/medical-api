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
import re

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
model = genai.GenerativeModel('gemini-1.5-flash')

@app.get("/healthz")
async def healthz(): return {"status": "ok"}

def clean_code(text):
    """Extrait uniquement le code Python entre les balises ou nettoie le texte"""
    if "```python" in text:
        text = re.search(r"```python (.*?)```", text, re.DOTALL).group(1)
    elif "```" in text:
        text = re.search(r"```(.*?)```", text, re.DOTALL).group(1)
    return text.strip().replace("```python", "").replace("```", "")

@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None)):
    try:
        file_bytes = await file.read()
        # On lit l'original
        df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
        df_mod = df_orig.copy()

        prompt = f"""
        Objet : DataFrame 'df'
        Colonnes : {list(df_orig.columns)}
        Action de Réginalde : "{instruction}"
        
        CONSIGNE : Écris uniquement le code Python pour modifier 'df'. 
        - Utilise df.at[index, 'colonne'] ou df.loc.
        - Ne réponds que par le code, sans explications.
        """
        
        response = model.generate_content(prompt)
        code = clean_code(response.text)

        # Exécution sécurisée
        exec_scope = {"df": df_mod, "pd": pd, "np": np}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        # Ré-injection dans Excel (Garde le Design)
        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        START_ROW = 5
        
        # On vide uniquement les données (pas le design)
        for row in ws.iter_rows(min_row=START_ROW, max_row=ws.max_row):
            for cell in row: cell.value = None

        # On remplit avec les nouvelles valeurs
        for r_idx, row_values in enumerate(df_mod.values):
            for c_idx, value in enumerate(row_values):
                ws.cell(row=START_ROW + r_idx, column=c_idx + 1).value = value

        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"Reginalde_{ts}.xlsx"
        fpath = os.path.join(HISTORY_DIR, fname)
        wb.save(fpath)
        return FileResponse(fpath, filename=fname)

    except Exception as e:
        print(f"ERREUR : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
        _, buffer = cv2.imencode('.jpg', enhanced)
        
        contents = [
            "Analyse ce document pour Réginalde (Tête de Coco). Extrait le texte de façon claire.",
            {"mime_type": "image/jpeg", "data": buffer.tobytes()}
        ]
        
        response = model.generate_content(contents)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        return {"text": response.text, "image": f"data:image/jpeg;base64,{b64_img}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
