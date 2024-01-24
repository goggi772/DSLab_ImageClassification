import os.path

import torchvision.transforms
from torchvision import datasets, transforms
from torchvision.io import read_image
import pandas as pd
from PIL import Image
import torch

from torch.utils.data import DataLoader, Dataset

module_transform = transforms.Compose([
    # transforms.Resize((1000, 469))
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.CenterCrop(224),
    # transforms.RandomCrop(32),
    # transforms.RandomRotation(degrees=30),
    transforms.ToTensor()
])
pre_trans = transforms.Compose([
    transforms.Resize((1000, 469)),
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, pre_trans, transform = None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.X = []

        for n in self.df["filename"]:
            img_path = os.path.join(self.img_dir, 'fault', n)
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, 'non_fault', n)
            self.X.append(pre_trans(Image.open(img_path)))
        self.y = self.df["label_re"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # img_name = self.df.iloc[idx, 0]
        # label = int(self.df.iloc[idx, 2])
        #
        # img_path = os.path.join(self.img_dir, 'fault', img_name)
        # if not os.path.exists(img_path):
        #     img_path = os.path.join(self.img_dir, 'non_fault', img_name)
        # # image = read_image(img_path)
        # image = Image.open(img_path)
        # if self.transform:
        #     image = self.transform(image)

        image, label = self.X[idx], self.y[idx]


        return image, torch.tensor(label)




def make_data_loader(args):
    train_dataset = CustomDataset(csv_file=args.trainCsv, img_dir=args.imgData, pre_trans=pre_trans, transform=module_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    valid_dataset = CustomDataset(csv_file=args.validCsv, img_dir=args.imgData, pre_trans=pre_trans, transform=module_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return train_loader, valid_loader

def make_test_loader(args):
    test_dataset = CustomDataset(csv_file=args.testCsv, img_dir=args.imgData, pre_trans=pre_trans, transform=module_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return test_loader


