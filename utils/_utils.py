import os.path
import cv2
import torchvision.transforms
from torchvision import datasets, transforms
from torchvision.io import read_image
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch


from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

torch.manual_seed(1004)

module_transform = transforms.Compose([
    transforms.Resize((1000, 469), antialias=True),
    transforms.RandomHorizontalFlip(),
])
test_transform = transforms.Compose([
    transforms.Resize((1000, 469), antialias=True),
])

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 클래스별 샘플 개수 계산
        self.class_counts = self.df['label_re'].value_counts().to_dict()

        # 클래스별 가중치 계산
        total_samples = len(self.df)
        self.weights = [total_samples / (len(self.class_counts) * count) for count in self.class_counts.values()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 2])

        img_path = os.path.join(self.img_dir, 'fault', img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, 'non_fault', img_name)

        image = self.transform(read_image(img_path).float())


        return image, label


    def get_weights(self):
        return [self.weights[label] for label in self.df['label_re']]


def make_data_loader(args):

    train_dataset = CustomDataset(csv_file=args.trainCsv, img_dir=args.imgData, transform=module_transform)

    class_weights = train_dataset.get_weights()
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=16, sampler=sampler)

    valid_dataset = CustomDataset(csv_file=args.validCsv, img_dir=args.imgData, transform=test_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    return train_loader, valid_loader

def make_test_loader(args):
    test_dataset = CustomDataset(csv_file=args.testCsv, img_dir=args.imgData, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    return test_loader

