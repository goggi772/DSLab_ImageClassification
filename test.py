import argparse
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from utils._utils import make_data_loader, make_test_loader
from model import BaseModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def test(args, test_data_loader, model):
    true = np.array([])
    pred = np.array([])

    model.eval()

    pbar = tqdm(test_data_loader)
    for i, (x, y) in enumerate(pbar):
        image = x.to(args.device)
        label = y.to(args.device)

        output = model(image)

        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy()
        pred = np.append(pred, output, axis=0)

        label = label.detach().cpu().numpy()
        true = np.append(true, label, axis=0)
    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--model-path', default='model/model_8.pth', help="Model's state_dict")
    parser.add_argument('--trainCsv', default='../../datasets/classification/train/re_data_df_first.csv', type=str,help='train label')
    parser.add_argument('--testCsv', default='../../datasets/classification/test/re_data_df_first.csv', type=str,help='test label')
    parser.add_argument('--validCsv', default='../../datasets/classification/valid/re_data_df_first.csv', type=str,help='valid label')
    parser.add_argument('--imgData', default='../../datasets/first/', type=str, help='img folder')
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device

    # hyperparameters
    args.batch_size = 4

    # Make Data loader and Model
    test_loader = make_test_loader(args)

    # instantiate model
    # model = BaseModel()
    # model = BaseModel()
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    print("======================")
    print("Test model: ", args.model_path)
    print("======================")


    # Test The Model
    pred, true = test(args, test_loader, model)

    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))
    print("Precision : {:.5f}".format(precision_score(true, pred)))
    print("Recall : {:.5f}".format(recall_score(true, pred)))
    print("F1-Score : {:.5f}".format(f1_score(true, pred)))
    print([(i, a, b) for i, (a, b) in enumerate(zip(pred, true)) if a != b])