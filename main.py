import os
import io
import time
import base64
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openpyxl import load_workbook
from groq import Groq

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Dossier pour l'historique
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/process")
async def process_medical_file(
    file: UploadFile = File(...),
    instruction: str = Form(None),
    audio: UploadFile = File(None)
):
    file_bytes = await file.read()
    df_original = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
    df_modified = df_original.copy()

    user_text = instruction
    if audio:
        audio_content = await audio.read()
        trans = client.audio.transcriptions.create(file=("a.wav", audio_content), model="whisper-large-v3", language="fr")
        user_text = trans.text

    prompt = f"Modifie le DataFrame 'df' (Colonnes: {list(df_original.columns)}). Instruction: '{user_text}'. Code pur uniquement."
    chat = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
    code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

    local_scope = {"df": df_modified, "pd": pd}
    exec(code, {}, local_scope)
    df_modified = local_scope["df"]

    wb = load_workbook(io.BytesIO(file_bytes))
    ws = wb.active
    for r_idx in range(len(df_original)):
        for col_idx in range(len(df_original.columns)):
            val_mod = df_modified.iat[row_idx, col_idx]
            ws.cell(row=5 + r_idx, column=col_idx + 1).value = val_mod

    # Sauvegarde dans l'historique avec un nom unique
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"Réginalde_{timestamp}.xlsx"
    filepath = os.path.join(HISTORY_DIR, filename)
    wb.save(filepath)
    
    return FileResponse(filepath, filename=filename)

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)
    return {"files": files}

@app.get("/download-history/{filename}")
async def download_history(filename: str):
    path = os.path.join(HISTORY_DIR, filename)
    return FileResponse(path)

@app.post("/ocr")
async def ocr_image(image: UploadFile = File(...)):
    """Analyse une photo de document et extrait le texte"""
    image_bytes = await image.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview", # Modèle Vision de Groq
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extrait tout le texte lisible de cette photo de document de suivi médical. Formate le texte pour qu'il soit facile à lire et à copier."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    )
    return {"text": completion.choices[0].message.content}

@app.get("/healthz")
async def healthz(): return {"status": "ok"}
