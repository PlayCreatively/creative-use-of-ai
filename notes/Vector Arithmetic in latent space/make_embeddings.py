# save as make_embeddings.py
# pip install torch transformers scikit-learn

from transformers import AutoTokenizer, AutoModel
import torch, json

model_names = ["gpt2", "./models/mistral7b"]

# Small, easy model for the demo. You can replace with a bigger one later.
MODEL_NAME = model_names[0]  # try "mistralai/Mistral-7B-v0.1" if you have VRAM
WORDS = [
    "king", "queen", "man", "woman", "paris", "france", "tokyo", "japan",
    "cat", "dog", "music", "art", "science", "philosophy", "computer"
]

tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True, output_hidden_states=True)
model.eval()

# load to GPU if available
# from transformers import AutoModel
# model = AutoModel.from_pretrained(
#     MODEL_PATH,
#     local_files_only=True,
#     torch_dtype="auto",
#     low_cpu_mem_usage=True
# )
# if torch.cuda.is_available():
# 	model.to("cuda")

with torch.no_grad():
    vecs = []
    labels = []
    for w in WORDS:
        # Tokenize as a standalone token sequence
        inputs = tok(w, return_tensors="pt")
        out = model(**inputs)
        # last hidden state: [batch, seq, hidden]
        h = out.last_hidden_state.squeeze(0)  # [seq, hidden]
        # mean-pool over tokens to handle multi-token words
        v = h.mean(dim=0).cpu().numpy()
        vecs.append(v)
        labels.append(w)

# PCA to 2D for visualization
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, random_state=0)
# xy = pca.fit_transform(vecs)

# pip install umap-learn
from umap import UMAP
reducer = UMAP(n_components=2, random_state=0)
xy = reducer.fit_transform(vecs)


# write JSON the p5 sketch expects
points = []
for (x, y), label in zip(xy, labels):
    points.append({
        "id": label,
        "label": label,
        "group": "words",
        "x": float(x),
        "y": float(y)
    })

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(points, f, ensure_ascii=False, indent=2)

print("Wrote embeddings.json with", len(points), "points.")
