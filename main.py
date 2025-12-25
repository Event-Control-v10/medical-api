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

# Autorisation CORS pour que le front puisse parler au back
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Dossier Historique
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/healthz")
async def healthz(): return {"status": "ok"}

# --- PARTIE EXCEL ---
@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None), audio: UploadFile = File(None)):
    file_bytes = await file.read()
    # On lit le fichier pour l'IA (on saute les 3 lignes de titre comme sur la photo)
    df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
    df_mod = df_orig.copy()

    # Gestion de l'audio si présent
    user_text = instruction
    if audio:
        audio_data = await audio.read()
        trans = client.audio.transcriptions.create(file=("a.wav", audio_data), model="whisper-large-v3", language="fr")
        user_text = trans.text

    # Demande de code à l'IA
    prompt = f"""
    Modifie le DataFrame 'df' (Colonnes: {list(df_orig.columns)}).
    Instruction: "{user_text}"
    RÈGLES: 
    1. Réponds UNIQUEMENT avec le code Python. 
    2. Utilise df.at[index, 'colonne'] pour être précis. 
    3. Pas de blabla, pas de markdown.
    """
    chat = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0)
    code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

    # Exécution
    exec_scope = {"df": df_mod, "pd": pd, "np": np}
    exec(code, {}, exec_scope)
    df_mod = exec_scope["df"]

    # Injection Openpyxl (Garde le Design)
    wb = load_workbook(io.BytesIO(file_bytes))
    ws = wb.active
    for r in range(len(df_orig)):
        for c in range(len(df_orig.columns)):
            ws.cell(row=5 + r, column=c + 1).value = df_mod.iat[r, c]

    # Sauvegarde Historique
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"Reginalde_{ts}.xlsx"
    fpath = os.path.join(HISTORY_DIR, fname)
    wb.save(fpath)
    return FileResponse(fpath, filename=fname)

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:10] # 10 derniers
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))

# --- PARTIE SCANNER (YEUX BIONIQUES) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    img_bytes = await image.read()
    
    # Amélioration de l'image (Contraste pour Réginalde)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    enhanced = cv2.convertScaleAbs(img, alpha=1.4, beta=10)
    _, buffer = cv2.imencode('.jpg', enhanced)
    b64_img = base64.b64encode(buffer).decode('utf-8')

    # Vision IA
    res = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Extrait le texte de ce document médical. Organise-le de façon très claire et aérée."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
        ]}]
    )
    return {"text": res.choices[0].message.content, "image": f"data:image/jpeg;base64,{b64_img}"}
