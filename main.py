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

# Configuration CORS pour GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossier pour l'historique des rapports
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# --- LOGIQUE DE MODIFICATION EXCEL ---
@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None), audio: UploadFile = File(None)):
    try:
        file_bytes = await file.read()
        
        # On lit les données (skiprows=3 pour passer les titres orange)
        df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
        df_mod = df_orig.copy()

        # Transcription de la voix si présente
        user_text = instruction
        if audio:
            audio_data = await audio.read()
            trans = client.audio.transcriptions.create(
                file=("a.wav", audio_data), 
                model="whisper-large-v3", 
                language="fr"
            )
            user_text = trans.text

        if not user_text:
            raise HTTPException(status_code=400, detail="Dis-moi quoi faire, Tête de Coco ! 🥥")

        # Demande de code Python à l'IA
        prompt = f"""
        DataFrame 'df' colonnes : {list(df_orig.columns)}
        Instruction de Réginalde : "{user_text}"
        
        RÈGLES CRITIQUES :
        - Écris UNIQUEMENT le code Python pour modifier 'df'.
        - Utilise df.at[index, 'colonne'] ou df.loc pour être précis.
        - Ne crée pas de nouvelles colonnes.
        - Pas de texte, pas de markdown (```).
        """
        
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        # Exécution de la modification
        exec_scope = {"df": df_mod, "pd": pd, "np": np}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        # Injection chirurgicale dans l'Excel ORIGINAL (Garde le Design)
        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        START_ROW = 5 # Les données commencent ligne 5 sur sa photo
        
        # On nettoie la zone de données avant d'écrire
        for row in ws.iter_rows(min_row=START_ROW, max_row=ws.max_row):
            for cell in row: cell.value = None

        # On remplit avec les données modifiées
        for r_idx, row_values in enumerate(df_mod.values):
            for c_idx, value in enumerate(row_values):
                ws.cell(row=START_ROW + r_idx, column=c_idx + 1).value = value

        # Sauvegarde pour l'historique
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"Reginalde_{ts}.xlsx"
        fpath = os.path.join(HISTORY_DIR, fname)
        wb.save(fpath)
        
        return FileResponse(fpath, filename=fname)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SCANNER BIONIQUE (OCR) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Amélioration du contraste pour les yeux de Réginalde
        enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
        _, buffer = cv2.imencode('.jpg', enhanced)
        b64_img = base64.b64encode(buffer).decode('utf-8')

        # Utilisation de Llama 3.2 Vision (Modèle 90B puissant)
        res = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyse ce document médical pour Réginalde. Extrait tout le texte lisible et organise-le proprement par rubriques."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }]
        )
        return {"text": res.choices[0].message.content, "image": f"data:image/jpeg;base64,{b64_img}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- GESTION HISTORIQUE ---
@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
