import os
import csv
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import *

os.makedirs(images_path, exist_ok=True)

def prepare_from_photos():

    classification = []
    label_index_metadata = []
    label_index = 0
    current_labels = []
    for folder in os.listdir(photos_path):
        folder_path = os.path.join(photos_path, folder)
        if os.path.isfile(folder_path):
            continue
        label = folder.replace("_cleaned", "")
        if "Others" in label:
            label = "Others"
        print("Label: ", label)
        if label not in current_labels:
            label_index_metadata.append((label_index, label))
            current_labels.append(label)
        else:
            label_index = [x[0] for x in label_index_metadata if x[1]==label][0]
        for image in tqdm(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image)
            output_image_path = os.path.join(images_path, folder+"_"+image)
            # shutil.copyfile(image_path, output_image_path)
            classification.append((image, label_index))
        label_index = label_index_metadata[-1][0] + 1


    print("Recording labels")
    with open(labels_path,'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['image','label'])
        for mytuple in classification: 
            csv_out.writerow(mytuple)

    with open(labels_metadata_path,'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['label_index','label'])
        for mytuple in label_index_metadata:
            csv_out.writerow(mytuple)
    print("Done")

def split_train_test(ratio = [0.8,0.1,0.1], seed=None):
    df = pd.read_csv(labels_path)
    print(df.head())
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(ratio[0] * m)
    validate_end = int(ratio[1] * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    train.to_csv(train_labels_path, index=False)
    validate.to_csv(val_labels_path, index=False)
    test.to_csv(test_labels_path, index=False)

prepare_from_photos()
split_train_test()
