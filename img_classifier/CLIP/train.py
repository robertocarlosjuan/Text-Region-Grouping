import os
import csv
import clip
import torch
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from config import *
from dataset import CustomDataset
from sklearn.metrics import classification_report

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the dataset
train = CustomDataset(images_path, [train_labels_path, val_labels_path], transform=preprocess)
test = CustomDataset(images_path, test_labels_path, transform=preprocess)

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

if not os.path.isfile(features_labels_file):
    # Calculate the image features
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)

    # Save image features
    features_labels = {"train_features": train_features, 
                        "train_labels": train_labels, 
                        "test_features": test_features, 
                        "test_labels": test_labels}
    with open(features_labels_file, "wb") as f:
        pickle.dump(features_labels, f)
else:
    # Load image features
    with open(features_labels_file, "rb") as f:
        features_labels = pickle.load(f)
    train_features = features_labels["train_features"]
    train_labels = features_labels["train_labels"]
    test_features = features_labels["test_features"]
    test_labels = features_labels["test_labels"]

if classifier_model == "logistic regression" and not os.path.isfile(model_paths[classifier_model]):
    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    with open(model_paths[classifier_model], "wb") as f:
        pickle.dump(classifier, f)
elif classifier_model == "mlp" and not os.path.isfile(model_paths[classifier_model]):
    # Perform MLP
    input_feature_size = train_features.shape[1]
    classifier = MLPClassifier(random_state=0, max_iter=1000, hidden_layer_sizes=(int(input_feature_size/2),), verbose=1)
    classifier.fit(train_features, train_labels)
    with open(model_paths[classifier_model], "wb") as f:
        pickle.dump(classifier, f)
else:
    model_path = model_paths[classifier_model]
    print("Loading pretrained model from %s" % (model_path))
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

# # Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
print(classification_report(test_labels, predictions, target_names=classes))
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print(f"Test Accuracy = {accuracy:.3f}")