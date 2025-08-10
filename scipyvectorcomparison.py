from openai import OpenAI
from pydub import AudioSegment, silence
import tempfile
from scipy.spatial.distance import cosine

client = OpenAI(api_key="<api-key>")

def remove_silence(audio_path, silence_thresh=-40, min_silence_len=1000):
    """
    Removes silent segments from audio using pydub.
    silence_thresh: dBFS below which is considered silence
    min_silence_len: duration of silence in ms
    """
    print(f"Removing silence from: {audio_path}...")
    audio = AudioSegment.from_file(audio_path)
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=100  # keep some context around silence boundaries
    )
    if not chunks:
        print("No speech detected, returning original audio.")
        return audio_path  # fallback
    cleaned_audio = AudioSegment.empty()
    for chunk in chunks:
        cleaned_audio += chunk
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    cleaned_audio.export(temp_file.name, format="wav")
    print("Removed silence")
    return temp_file.name

def transcribe_audio(file_path):
    print("Transcribing audio...")
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    print(f"Transcript: {transcript.text}")
    print("Transcribed audio")

    return transcript.text

def get_text_embedding(text, model="text-embedding-3-small"):
    print("Generating embedding...")
    response = client.embeddings.create(
        model=model,
        input=text
    )
    embedding = response["data"][0]["embedding"]
    print("Generated embedding")
    return embedding

def process_audio_to_embedding(audio_path):
    print("Process audio to embedding...")
    cleaned_path = remove_silence(audio_path)
    text = transcribe_audio(cleaned_path)
    embedding = get_text_embedding(text)
    print("Processed audio to embedding")
    return embedding

def compare_audio_embeddings(audio1, audio2):
    print("Compare audio embeddings with SciPy cosine similarity...")
    emb1 = process_audio_to_embedding(audio1)
    emb2 = process_audio_to_embedding(audio2)
    similarity = 1 - cosine(emb1, emb2)  # cosine returns distance, subtract from 1
    print(f"Cosine similarity: {similarity:.6f}")
    print("Compared audio embeddings")
    return similarity

# Example usage:
# if __name__ == "__main__":
#     similarity = compare_audio_embeddings("audio1.wav", "audio2.wav")
#     print(f"Similarity score: {similarity}")
