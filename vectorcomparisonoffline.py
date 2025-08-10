import os
import tempfile
from pydub import AudioSegment, silence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

# Load offline models once
print("Loading offline transcription model...")
#whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
whisper_model = WhisperModel(r"C:\Users\STSadanandan\models\WhisperModel", device="cpu", compute_type="int8")

print("Loading offline embedding model...")

embedding_model = SentenceTransformer(r"C:\Users\STSadanandan\models\all-MiniLM-L6-v2\git")
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def remove_silence(input_audio_path, silence_thresh=-40, min_silence_len=500):
    """Removes silence from audio file and returns path to processed file."""
    sound = AudioSegment.from_file(input_audio_path)
    chunks = silence.split_on_silence(sound, 
                                      min_silence_len=min_silence_len,
                                      silence_thresh=silence_thresh)
    processed = AudioSegment.empty()
    for chunk in chunks:
        processed += chunk
    temp_path = tempfile.mktemp(suffix=".wav")
    processed.export(temp_path, format="wav")
    return temp_path

def transcribe_audio(file_path):
    """Transcribes audio locally using faster-whisper."""
    segments, info = whisper_model.transcribe(file_path)
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def get_text_embedding(text):
    """Generates local embedding for text."""
    return embedding_model.encode([text])[0].reshape(1, -1)

def process_audio_to_embedding(file_path):
    """Removes silence, transcribes, and generates embedding."""
    cleaned_path = remove_silence(file_path)
    transcript = transcribe_audio(cleaned_path)
    print(f"transcript = {transcript}")
    embedding = get_text_embedding(transcript)
    return embedding

def compare_audio_embeddingswithfiles(file1, file2):
    """Compares two audio files based on text embeddings."""
    emb1 = process_audio_to_embedding(file1)
    emb2 = process_audio_to_embedding(file2)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def compare_audio_embeddings(embedding1, embedding2):
    """Compares two audio files based on text embeddings."""
    # emb1 = process_audio_to_embedding(file1)
    # emb2 = process_audio_to_embedding(file2)
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)

    # print(f"embedding1: {embedding1}, shape: {np.array(embedding1).shape if embedding1 is not None else None}")
    # print(f"embedding2: {embedding2}, shape: {np.array(embedding2).shape if embedding2 is not None else None}")

    if embedding1 is None or embedding2 is None or len(embedding1) == 0 or len(embedding2) == 0:
        return {"error": "One or both embeddings are empty or not found"}
    # Check if embeddings are empty
    if emb1.size == 0 or emb2.size == 0:
        raise ValueError("One or both embeddings are empty")

    # Reshape to 2D: 1 sample, N features
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)

    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity