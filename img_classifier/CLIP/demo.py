import os
import torch
import clip
import pickle
from glob import glob
from PIL import Image
from img_classifier.CLIP.config import *

def classify(sample_image_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    #model.to(device)

    if os.path.isfile(sample_image_path):
        images_path_list = [sample_image_path]
    elif os.path.isdir(sample_image_path):
        images_path_list = glob('{}/*.jpg'.format(sample_image_path))#[os.path.join(sample_image_path, x) for x in os.listdir(sample_image_path)]
        print("Predicting images from %s" % sample_image_path)

    model_path = os.path.join(os.path.abspath(os.curdir), model_paths[classifier_model])
    print("Loading pretrained model from %s" % (model_path))
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    
    images = [preprocess(Image.open(sample_image)).unsqueeze(0).to(device) for sample_image in images_path_list]

    results = []
    with torch.no_grad():
        for image, img_path in zip(images, images_path_list):
            image_features = model.encode_image(image)
            probs=classifier.predict_proba(image_features.cpu())
#            probs = classifier.predict_proba(torch.tensor(image_features))
            # Sort predicted classes in descending order of probability
            class_probs = sorted([(p, classes[i]) for i, p in enumerate(probs[0])], reverse=True)
            image_path_basename = os.path.basename(img_path)
            label = class_probs[0][1]
            results.append({'filename': image_path_basename, 'label': label})
            # os.makedirs('{}/{}'.format(sample_image_path, label), exist_ok=True)
            # os.rename('{}/{}'.format(sample_image_path, image_path_basename), "{}/{}/{}".format(sample_image_path, label, image_path_basename))
            # print("\nPredicted Classes for %s" % image_path_basename)
            # print("\n".join(["%s %.3f" % (x[1], x[0]) for x in class_probs]))

    return results