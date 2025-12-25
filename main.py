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

# Configuration CORS pour autoriser ton site GitHub
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossier pour garder les fichiers modifiés (Historique)
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/process")
async def process_excel(file: UploadFile = File(...), instruction: str = Form(None), audio: UploadFile = File(None)):
    try:
        file_bytes = await file.read()
        # On lit l'Excel (skiprows=3 car les données commencent après les titres)
        df_orig = pd.read_excel(io.BytesIO(file_bytes), skiprows=3)
        df_mod = df_orig.copy()

        # 1. Gestion de l'audio (Whisper)
        user_text = instruction
        if audio:
            audio_data = await audio.read()
            trans = client.audio.transcriptions.create(file=("a.wav", audio_data), model="whisper-large-v3", language="fr")
            user_text = trans.text

        if not user_text:
            return {"error": "Aucune instruction reçue"}

        # 2. IA Llama 3.3 pour générer le code de modification
        prompt = f"""
        Modifie le DataFrame 'df' (Colonnes: {list(df_orig.columns)}).
        Instruction: "{user_text}"
        RÈGLES: 
        1. Réponds UNIQUEMENT avec le code Python. 
        2. Utilise df.at, df.loc ou df.drop pour modifier précisément. 
        3. Pas de blabla, pas de markdown.
        """
        chat = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0)
        code = chat.choices[0].message.content.strip().replace("```python", "").replace("```", "")

        # 3. Exécution du code généré
        exec_scope = {"df": df_mod, "pd": pd, "np": np}
        exec(code, {}, exec_scope)
        df_mod = exec_scope["df"]

        # 4. Injection CHIRURGICALE (Openpyxl) - GARDE LE DESIGN
        wb = load_workbook(io.BytesIO(file_bytes))
        ws = wb.active
        
        # --- LOGIQUE ROBUSTE : On vide l'ancienne zone et on remplit la nouvelle ---
        START_ROW = 5 # Les données commencent à la ligne 5 dans ton fichier
        
        # On vide les anciennes valeurs pour éviter les "lignes fantômes"
        for row in ws.iter_rows(min_row=START_ROW, max_row=ws.max_row):
            for cell in row:
                cell.value = None

        # On injecte les nouvelles données du DataFrame modifié
        for r_idx, row_values in enumerate(df_mod.values):
            for c_idx, value in enumerate(row_values):
                # On écrit la valeur (r_idx + START_ROW car Excel commence à 1)
                ws.cell(row=START_ROW + r_idx, column=c_idx + 1).value = value

        # 5. Sauvegarde
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"Reginalde_{ts}.xlsx"
        fpath = os.path.join(HISTORY_DIR, fname)
        wb.save(fpath)
        
        return FileResponse(fpath, filename=fname)

    except Exception as e:
        print(f"Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

        res = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Extrait le texte de ce document médical. Organise-le de façon très claire par rubriques pour que ce soit facile à lire."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}]
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
