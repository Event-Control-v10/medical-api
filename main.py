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
import google.generativeai as genai
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

# Clients
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- CORRECTION ICI : ON UTILISE LE NOM STANDARD ---
model_vision = genai.GenerativeModel('gemini-1.5-flash')

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
            trans = client_groq.audio.transcriptions.create(file=("a.wav", audio_data), model="whisper-large-v3", language="fr")
            user_text = trans.text

        if not user_text: return {"error": "Aucune consigne"}

        prompt = f"""
        DataFrame df (Colonnes: {list(df_orig.columns)}). Instruction: "{user_text}".
        RÈGLES ABSOLUES:
        1. Code Python uniquement.
        2. Utilise df.at[index, 'col'] = val.
        3. NE TOUCHE PAS aux lignes non concernées.
        4. Pas de blabla.
        """
        
        chat = client_groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0)
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        exec_scope = {"df": df_mod, "pd": pd}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        START_ROW = 5 
        
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

# --- SCANNER ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        print(f"Réception image : {image.filename}")
        img_bytes = await image.read()
        
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # On garde une taille correcte pour la lisibilité
        img.thumbnail((1024, 1024))
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        image_data = buffered.getvalue()

        print("Envoi à Gemini 1.5 Flash...")
        response = model_vision.generate_content([
            "Analyse ce document médical. Extrait tout le texte lisible. Sois précis et structuré.",
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return {
            "text": response.text,
            "image": f"data:image/jpeg;base64,{base64_image}"
        }

    except Exception as e:
        print(f"ERREUR CRITIQUE OCR : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur technique : {str(e)}")

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
