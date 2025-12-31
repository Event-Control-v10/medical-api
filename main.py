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

# --- MODÈLES GROQ ---
# Pour le texte (Cerveau) : On garde Llama 3.3 (Le meilleur)
MODEL_EXCEL = "llama-3.3-70b-versatile"

# Pour la vision (Yeux) : On passe sur Llava (Le seul survivant)
MODEL_VISION = "llava-v1.5-7b-4096-preview"

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

        if not user_text: return {"error": "Aucune consigne"}

        prompt = f"""
        DataFrame df colonnes: {list(df_orig.columns)}. 
        Instruction: "{user_text}". 
        RÈGLES: 
        - Code Python uniquement. 
        - Utilise df.at[index, 'col'] = val.
        - Pas de markdown.
        """
        
        chat = client.chat.completions.create(model=MODEL_EXCEL, messages=[{"role": "user", "content": prompt}], temperature=0)
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        exec_scope = {"df": df_mod, "pd": pd}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

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

# --- PARTIE SCANNER (LLAVA) ---
@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        
        # Préparation image
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Llava a besoin d'images petites
        img.thumbnail((800, 800))
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{base64_image}"

        # Appel à Llava (Le modèle de secours)
        response = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Liste tout le texte visible sur cette image médicale. Sois direct."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": data_url}
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        return {
            "text": response.choices[0].message.content,
            "image": data_url
        }
    except Exception as e:
        print(f"DEBUG GROQ : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur Groq: {str(e)}")

@app.get("/history")
async def get_history():
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)[:15]
    return {"files": files}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(HISTORY_DIR, filename))
