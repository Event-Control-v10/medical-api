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

# Configuration pour autoriser GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR): os.makedirs(HISTORY_DIR)

# --- CLIENT 1 : GROQ (Pour la logique et le code) ---
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CLIENT 2 : GOOGLE GEMINI (Pour les yeux bioniques) ---
# Si tu n'as pas encore la clé, va sur https://aistudio.google.com/app/apikey
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model_vision = genai.GenerativeModel('gemini-1.5-flash')

@app.get("/healthz")
async def healthz(): return {"status": "ok"}

# --- PARTIE EXCEL (Cerveau: Groq) ---
@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None), audio: UploadFile = File(None)):
    try:
        file_bytes = await file.read()
        df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
        df_mod = df_orig.copy()

        user_text = instruction
        # Transcription Audio via Groq (Ultra rapide)
        if audio:
            audio_data = await audio.read()
            trans = client_groq.audio.transcriptions.create(file=("a.wav", audio_data), model="whisper-large-v3", language="fr")
            user_text = trans.text

        if not user_text: return {"error": "Aucune consigne"}

        # Génération du code Python via Groq Llama 3.3
        prompt = f"""
        DataFrame 'df' colonnes: {list(df_orig.columns)}. 
        Instruction: "{user_text}". 
        RÈGLES: 
        - Code Python uniquement. 
        - Utilise df.at[index, 'col'] = val.
        - Pas de markdown.
        """
        chat = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}], 
            temperature=0
        )
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        # Exécution
        exec_scope = {"df": df_mod, "pd": pd}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        # Injection dans l'Excel original (Garde le Design)
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

# --- PARTIE SCANNER (Yeux: Gemini) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        
        # Nettoyage de l'image (Fix RGBA/Palette)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        
        # Analyse par Gemini Vision
        response = model_vision.generate_content([
            "Tu es un assistant médical. Analyse ce document pour Réginalde. Extrait tout le texte et structure-le.",
            {"mime_type": "image/jpeg", "data": buffered.getvalue()}
        ])
        
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "text": response.text,
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
