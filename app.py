from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from sqllite import create_database, store_embedding, load_embeddings
from vectorcomparisonoffline import process_audio_to_embedding, remove_silence, transcribe_audio, get_text_embedding, compare_audio_embeddings
from slidingwindowcomparison import get_sliding_window_embeddings, compare_sliding_embeddings
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
app = FastAPI()
# Allow all origins (or restrict to specific ones)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_database()

UPLOAD_FOLDER = "audio_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/save")
async def save_audio(file: UploadFile = File(...), song_id: str = Form(...)):
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)


    embedding = process_audio_to_embedding(path)
    #store_embedding(embedding, song_id=song_id)
    store_embedding(song_id, embedding, db_path="embeddings.db")
    return {"message": "embedding completed", "embedding": len(embedding)}

@app.post("/match")
async def match_audio(file: UploadFile = File(...), song_id: str = Form(...)):
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    embedding1 = process_audio_to_embedding(path)
    embedding2 = load_embeddings(song_id, db_path="embeddings.db")

    match = compare_audio_embeddings(embedding1, embedding2)
    return {"match": match}


# Example usage:
@app.post("/slidingaudiomatch")
async def match_slidingaudio(file: UploadFile = File(...),file2: UploadFile = File(...), song_id: str = Form(...)):
    audio_path_1 = os.path.join(UPLOAD_FOLDER, file.filename)
    # audio_path_1 = "conversation_full.wav"
    audio_path_2 = os.path.join(UPLOAD_FOLDER, file2.filename)

    window_size = 3.0  # seconds
    stride = 1.0       # seconds

    # Get embeddings for sliding windows
    embeddings1 = get_sliding_window_embeddings(audio_path_1, window_size, stride)
    embeddings2 = get_sliding_window_embeddings(audio_path_2, window_size, stride)

    # Compare embeddings using sliding window approach
    match_score = compare_sliding_embeddings(embeddings1, embeddings2)

    print(f"Sliding window match score: {match_score:.4f}")
    #return match_score
    return {"match score": float(match_score)}


# @app.post("/match")
# async def match_audio(file: UploadFile = File(...), song_id: str = Form(...)):
#     path = os.path.join(UPLOAD_FOLDER, file.filename)
#     with open(path, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     embedding1 = process_audio_to_embedding(path)
#     embedding2 = load_embeddings(song_id, db_path="embeddings.db")  # Assuming you want to match against the first stored embedding id, db_path="embeddings.db"
#     match = compare_audio_embeddings(embedding1, embedding2)  # Assuming embedding2 is provided elsewhere
#     return {"match": match[0], "score": match[1]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # <filename>:<FastAPI instance>
        host="127.0.0.1",
        port=8000,
        reload=True
    )