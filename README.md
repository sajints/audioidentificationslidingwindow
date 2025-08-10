
# Encoder
## Convert m4a to mp3
ffmpeg -i WhisperAPI.m4a -c:v copy -c:a libmp3lame -q:a 4 WhisperAPI.mp3
ffmpeg -i TorchNumpy.m4a -c:v copy -c:a libmp3lame -q:a 4 TorchNumpy.mp3
ffmpeg -i marunadan.m4a -c:v copy -c:a libmp3lame -q:a 4 marunadan.mp3
ffmpeg -i sandra1.m4a -c:v copy -c:a libmp3lame -q:a 4 sandra1.mp3



## Cut audio
ffmpeg -i WhisperAPI.mp3 -ss 0 -to 10 -c copy part1.mp3
ffmpeg -i WhisperAPI.mp3 -ss 20 -c copy part2.mp3
echo file 'part1.mp3' > file_list.txt
echo file 'part2.mp3' >> file_list.txt
ffmpeg -f concat -safe 0 -i file_list.txt -c copy output.mp3



# OFFLINE downloads
How to fix for SentenceTransformer?
Option 1: Download SentenceTransformer model manually and load locally
Go to the model page:

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Download the entire folder (all files: config.json, pytorch_model.bin, tokenizer.json, etc).

Put them into a local folder, e.g.:

makefile
Copy
Edit
C:\Users\STSadanandan\models\all-MiniLM-L6-v2\
Then load like this in Python:

python
Copy
Edit
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(r"C:\Users\STSadanandan\models\all-MiniLM-L6-v2")



# SIMILARITY SCORE

Fullaudio(49 seconds) with Part1(first 10 seconds)                              - "match score": 0.9860297441482544
Fullaudio with muted 10 seconds audio                                           - "match score": 0.9997168183326721
Fullaudio with Part2(last 29 seconds)                                           - "match score": 0.9885079860687256
Full audio with audio combining Part1 and Part2 into single audio(39 seconds)   - "match score": 0.9860297441482544
Fullaudio with me reading another content                                       - "match score": 0.7339223623275757
Fullaudio with another conversation 1 recorded from youtube                     - "match score": 0.2570127248764038
Fullaudio with another conversation 2 recorded from youtube                     - "match score": 0.28087976574897766
Fullaudio with another conversation 3 recorded from youtube                     - "match score": 0.29200059175491333
Fullaudio with another conversation 4 recorded from youtube                     - "match score": 0.2967171370983124
Part1 with conversation 4 recorded from youtube                                 - "match score": 0.25557011365890503
conversation 3 & 4 recorded from youtube(different parts of same video)         - "match score": 0.5774316191673279


Cosine Similarity Score	Interpretation	Likelihood of Same Audio
0.85 – 1.0	Very high similarity	Highly likely
0.70 – 0.85	Moderate to high similarity	Likely
0.50 – 0.70	Moderate similarity	Possible
0.30 – 0.50	Low similarity	Unlikely
< 0.30	Very low or no similarity	Very unlikely


Results:
Full audio - 49secs is saved to DB with ID=1
part1 - first 10 secs, comparison result =1
part2 - last 29 secs, result=0.66898480385383526


