# pip install sentence-transformers numpy
from sentence_transformers import SentenceTransformer
import numpy as np



transformers = [
    "sentence-transformers/all-mpnet-base-v2", # BERT-based model
    "intfloat/e5-small-v2",
    "sentence-transformers/all-MiniLM-L6-v2" # smaller model
    ]

# load a small open embedding model
model = SentenceTransformer(transformers[0])

# define your words
words = ["king", "queen", "man", "woman", "prince", "princess",
         "city", "village", "cat", "dog", "doctor", "nurse", "medical", "hospital", "job", "human"]

f = open("demofile.txt")

words = f.read().splitlines()

# encode & normalize
embeds = model.encode(words, normalize_embeddings=True)
word2vec = dict(zip(words, embeds))

female_vector = word2vec["woman"] - word2vec["man"]
male_vector = -female_vector

# vector arithmetic
# target = word2vec["king"] - word2vec["man"] + word2vec["woman"]
target = (word2vec["doctor"] - word2vec["nurse"]) + word2vec["human"]
target /= np.linalg.norm(target)

# compute cosine similarity
def cosine(a, b): return np.dot(a, b)
scores = {w: cosine(target, v) for w, v in word2vec.items() if w not in ["doctor", "nurse", "human"]}
for w, s in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"{w:10s} {s:.3f}")
