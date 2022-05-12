from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
from config import classes
device = ("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        if type(annotation_file) == str:
            self.annotations = pd.read_csv(annotation_file)
        else:
            self.annotations = pd.concat([pd.read_csv(annot) for annot in annotation_file], ignore_index=True)
            print("Length of dataset: ", len(self.annotations))
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        y_label = self.annotations.iloc[index, 1]
        if not img_id.isnumeric():
            y_label = classes.index(y_label)
        y_label = torch.tensor(float(y_label))

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)