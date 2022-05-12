import os
import csv

classifier_model = "mlp"
sample_image_path = "shot_detection/shotRlt"

clip_path = "img_classifier/CLIP/"
data_path = os.path.join(clip_path, "data")
images_path = os.path.join(data_path, "images")
features_labels_file = os.path.join(data_path, "features_labels.pkl")
weights_path = os.path.join(clip_path, "weights")
os.makedirs(weights_path, exist_ok=True)
model_paths = {
    "logistic regression" : os.path.join(weights_path, "log_reg_model.pkl"),
    "mlp" : os.path.join(weights_path, "mlp_model.pkl")
}

train_labels_path = os.path.join(data_path, 'train.csv')
val_labels_path = os.path.join(data_path, 'val.csv')
test_labels_path = os.path.join(data_path, 'test.csv')

photos_path = os.path.join(data_path, "photos")
labels_path = os.path.join(data_path, 'labels.csv')
labels_metadata_path = os.path.join(data_path, 'labels_metadata.csv')

# Labels
classes=[]         #an empty list to store the second column
with open(labels_metadata_path, 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for i, row in enumerate(reader):
        if i==0:
            continue
        classes.append(row[1])