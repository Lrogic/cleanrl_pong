import torch
import clip
from PIL import Image

import time

t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("dog.png")).unsqueeze(0).to(device)
print(image.shape)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

for _ in range(100):
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

t1 = time.time()
total = t1-t0

print(f"Total time for 100 iterations: {total:.2f} seconds")