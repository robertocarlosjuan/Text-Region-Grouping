import torch
import clip
import pickle
from PIL import Image
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open(sample_image_path)).unsqueeze(0).to(device)

model_path = model_paths[classifier_model]
print("Loading pretrained model from %s" % (model_path))
with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

with torch.no_grad():
    image_features = model.encode_image(image)
    probs = classifier.predict_proba(image_features)

# Sort predicted classes in descending order of probability
class_probs = sorted([(p, classes[i]) for i, p in enumerate(probs[0])], reverse=True)
print("Predicted Classes")
print("\n".join(["%s %.3f" % (x[1], x[0]) for x in class_probs]))
