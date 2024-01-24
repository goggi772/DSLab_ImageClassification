import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from model import BaseModel
from utils._utils import make_data_loader
from torchvision.models import resnet18, ResNet18_Weights



def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()


def train(args, train_data_loader, val_data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    for epoch in range(args.epochs):
        train_losses = []
        train_acc = 0.0
        total = 0
        print(f"[Epoch {epoch + 1} / {args.epochs}]")

        model.train()
        pbar = tqdm(train_data_loader)
        for i ,(x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)
            optimizer.zero_grad()

            output = model(image)

            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        model.eval()
        val_loss = []
        val_acc = 0.0
        val_total = 0



        pbar2 = tqdm(val_data_loader)
        with torch.no_grad():
            for j, (images, labels) in enumerate(pbar2):
                images = images.to(args.device)
                labels = labels.to(args.device)

                output = model(images)

                loss = criterion(output, labels)

                val_loss.append(loss.item())
                val_total += (labels.size(0))

                val_acc += acc(output, labels)


        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc / total

        epoch_val_loss = np.mean(val_loss)
        epoch_val_acc = val_acc / val_total

        print(f'Epoch {epoch + 1}')
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc * 100))

        print(f'val_loss : {epoch_val_loss}')
        print('val_accuracy : {:.3f}'.format(epoch_val_acc * 100))

        torch.save(model.state_dict(), f'{args.save_path}/model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Classification')
    parser.add_argument('--save-path', default='model/', help="Model's state_dict")
    parser.add_argument('--trainCsv', default='../../datasets/classification/train/re_data_df_first.csv', type=str, help='train label')
    parser.add_argument('--testCsv', default='../../datasets/classification/test/re_data_df_first.csv', type=str, help='test label')
    parser.add_argument('--validCsv', default='../../datasets/classification/valid/re_data_df_first.csv', type=str, help='valid label')
    parser.add_argument('--imgData', default='../../datasets/first/', type=str, help='img folder')

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    args.device = device

    # hyperparameters
    args.epochs = 5
    args.learning_rate = 0.0001
    args.batch_size = 16

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())

    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")

    # Make Data loader and Model
    train_loader, valid_loader = make_data_loader(args)

    # custom model
    # model = BaseModel()

    # torchvision model
    # rn18 = resnet18(weights=ResNet18_Weights.DEFAULT)
    #
    # rn18.fc.out_features=2
    # rn18.to(device)
    # rn18.fc = nn.Linear(rn18.fc.in_features, 10)
    # rn18 = rn18.to(device)

    # print(rn18)

    model = BaseModel()
    model = model.to(device)

    # Training The Model
    train(args, train_loader, valid_loader, model)
