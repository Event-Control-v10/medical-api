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

# Configuration CORS pour autoriser ton Front-end GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossier Historique
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# --- PARTIE EXCEL CHIRURGICALE ---
@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None), audio: UploadFile = File(None)):
    try:
        file_bytes = await file.read()
        
        # Lecture propre : on détecte les colonnes à la ligne 4 (index 3)
        df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
        df_mod = df_orig.copy()

        # Transcription audio si nécessaire
        user_text = instruction
        if audio:
            audio_data = await audio.read()
            trans = client.audio.transcriptions.create(
                file=("audio.wav", audio_data), 
                model="whisper-large-v3", 
                language="fr"
            )
            user_text = trans.text

        if not user_text:
            raise HTTPException(status_code=400, detail="Aucune instruction")

        # Prompt ultra-précis pour éviter de casser la structure
        prompt = f"""
        Tu es un expert Pandas. Modifie le DataFrame 'df'.
        Colonnes disponibles : {list(df_orig.columns)}
        Instruction de Réginalde : "{user_text}"
        
        RÈGLES :
        1. Utilise 'df.at[index, "colonne"]' ou 'df.loc'.
        2. NE SUPPRIME PAS de colonnes.
        3. NE MODIFIE QUE ce qui est demandé.
        4. Réponds UNIQUEMENT avec le code Python, pas de blabla.
        """
        
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        # Exécution sécurisée
        exec_scope = {"df": df_mod, "pd": pd, "np": np}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        # Injection dans l'Excel original (Protection du design)
        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        
        # On utilise len(df_mod) pour éviter l'IndexError
        # On écrit les données à partir de la ligne 5 d'Excel
        for r in range(len(df_mod)):
            for c in range(len(df_mod.columns)):
                # row=5 car ligne 1-3=titre, 4=en-têtes, 5=données
                ws.cell(row=5 + r, column=c + 1).value = df_mod.iat[r, c]

        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"Reginalde_{ts}.xlsx"
        fpath = os.path.join(HISTORY_DIR, fname)
        wb.save(fpath)
        
        return FileResponse(fpath, filename=fname)
    
    except Exception as e:
        print(f"ERREUR : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- PARTIE SCANNER (YEUX BIONIQUES) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Augmentation du contraste pour Réginalde
        enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
        _, buffer = cv2.imencode('.jpg', enhanced)
        b64_img = base64.b64encode(buffer).decode('utf-8')

        res = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Extrait le texte de ce document médical. Sois très précis et organise les informations par rubriques."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }]
        )
        return {"text": res.choices[0].message.content, "image": f"data:image/jpeg;base64,{b64_img}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
