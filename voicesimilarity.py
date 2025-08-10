# import librosa
# import numpy as np
# import torch
# from pyannote.audio import Model
# from pyannote.audio.pipelines.utils.hook import ProgressHook
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# # Load a pretrained speaker embedding model
# embedding_model = PretrainedSpeakerEmbedding(
#     "speechbrain/spkrec-ecapa-voxceleb",  # high-quality speaker embeddings
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )

# def process_audio_to_embedding(audio_path, start_sec, window_size_sec):
#     """
#     Extracts an embedding for a specific segment of an audio file.
#     - audio_path: path to audio file
#     - start_sec: segment start time in seconds
#     - window_size_sec: segment length in seconds
#     Returns: numpy array (1, embedding_dim)
#     """
#     # Load the segment
#     y, sr = librosa.load(audio_path, sr=16000, offset=start_sec, duration=window_size_sec)
    
#     # Convert to embedding (expects tensor [batch, time])
#     embedding = embedding_model(torch.tensor(y).unsqueeze(0))
    
#     # Convert to numpy
#     return embedding.detach().cpu().numpy()

import librosa
import torch
import numpy
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

print(torch.__version__)
print(numpy.__version__)

def process_audio_to_embedding(audio_path, start_sec, window_size_sec):
    y, sr = librosa.load(audio_path, sr=16000, offset=start_sec, duration=window_size_sec)
    #signal = torch.tensor(y).unsqueeze(0)
    signal = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    embedding = classifier.encode_batch(signal)
    return embedding.detach().cpu().numpy()
