import openai
import os

# Set your OpenAI API key
openai.api_key = "<api key here>" #os.getenv("OPENAI_API_KEY")

def transcribe_audio(file_path):
    """
    Transcribes speech from an audio file using OpenAI Whisper API.
    """
    print("Transcribing audio...")
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

def get_text_embedding(text, model="text-embedding-3-small"):
    """
    Generates a text embedding from transcribed text.
    """
    print("Generating embedding...")
    response = openai.Embedding.create(
        model=model,
        input=text
    )
    return response["data"][0]["embedding"]

def process_audio_to_embedding(file_path):
    """
    Complete process: audio → text → embedding vector
    """
    text = transcribe_audio(file_path)
    print(f"Transcript: {text}")
    embedding = get_text_embedding(text)
    print(f"Embedding vector length: {len(embedding)}")
    return embedding

# Example usage
# if __name__ == "__main__":
#     audio_path = "example_audio.wav"  # Replace with your file path
#     embedding_vector = process_audio_to_embedding(audio_path)