import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from voicesimilarity import process_audio_to_embedding

def get_sliding_window_embeddings(audio_path, window_size=3.0, stride=1.0, total_duration=None):
    """
    Splits audio into windows and returns embeddings for each window.
    - audio_path: path to audio file
    - window_size: window length in seconds
    - stride: step size between windows in seconds
    - total_duration: total audio length in seconds (if None, calculate)
    
    Returns: list of embeddings (numpy arrays)
    """
    # For example, you can get total duration via librosa or other audio lib
    if total_duration is None:
        import librosa
        total_duration = librosa.get_duration(filename=audio_path)
    
    embeddings = []
    start = 0.0
    
    while start + window_size <= total_duration:
        emb = process_audio_to_embedding(audio_path, start, window_size)
        embeddings.append(emb)
        start += stride
    
    return embeddings

def compare_sliding_embeddings(embeddings1, embeddings2):
    """
    Compare two lists of embeddings and return the max cosine similarity.

    This function flattens any extra dimensions in the embeddings so that
    cosine_similarity receives 2D arrays of shape (num_windows, embedding_dim).
    """
    if not embeddings1 or not embeddings2:
        return 0.0

    # Convert embeddings to numpy arrays and flatten each embedding to 1D
    emb1_flattened = [emb.reshape(-1) for emb in embeddings1]
    emb2_flattened = [emb.reshape(-1) for emb in embeddings2]

    # Stack embeddings into 2D arrays (num_windows, features)
    emb1_array = np.vstack(emb1_flattened)
    emb2_array = np.vstack(emb2_flattened)

    # Compute cosine similarity matrix between all window embeddings
    sim_matrix = cosine_similarity(emb1_array, emb2_array)

    # Return the maximum similarity value among all pairs
    max_sim = np.max(sim_matrix)

    return max_sim

# # Example usage:

# audio_path_1 = "conversation_full.wav"
# audio_path_2 = "conversation_clip.wav"

# window_size = 3.0  # seconds
# stride = 1.0       # seconds

# # Get embeddings for sliding windows
# embeddings1 = get_sliding_window_embeddings(audio_path_1, window_size, stride)
# embeddings2 = get_sliding_window_embeddings(audio_path_2, window_size, stride)

# # Compare embeddings using sliding window approach
# match_score = compare_sliding_embeddings(embeddings1, embeddings2)

# print(f"Sliding window match score: {match_score:.4f}")
