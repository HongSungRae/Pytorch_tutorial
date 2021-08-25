from numpy import datetime_data
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from model import NeuralNetwork
import os
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from utils import AverageMeter
import torch.nn as nn
from torchmetrics.functional import accuracy



parser = argparse.ArgumentParser('Test model')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='GPU device')
parser.add_argument('--data_path', default='./data', type=str,
                    help='data path')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


# model = NeuralNetwork().cuda()
# model = nn.DataParallel(model).cuda()
# model.load_state_dict(torch.load(args.save_path+'/my_network_5.pth'))
# model.eval()



def visualize(model, test_dataset):
    '''
    Sample 9 random data and
    visualize GT & predicted label
    '''
    labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
        img, _ = test_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        pred = torch.argmax(model(img),dim=-1).item()
        plt.title('Prediction :'+labels_map[pred])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()




def test():
    '''
    For classification : Accuracy
    Criterion : nn.CrossEntropyLoss()
    '''
    
    # init settings
    model = NeuralNetwork().cuda()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.save_path+'/my_network_5.pth'))
    model.eval()

    val_dataset = datasets.FashionMNIST(
        root=args.data_path,
        train=False,
        download=True,
        transform=ToTensor(),
        )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    

    # evaluation
    acc = AverageMeter()
    with torch.no_grad():
        for i,(data,label) in enumerate(val_loader):
            data, label = data.cuda(), label.cuda()
            pred = model(data)
            acc.update(accuracy(pred,label).item(),n=data.shape[0])
            if (i+1)%10==0:
                print(f'Now testing on : [{i+1}/{len(val_loader)}] | Acc : [{acc.avg}]')
    print(f'Avg Accuracy : [{acc.avg}]')
    print(f'Accuracy std : [{acc.std}]')
    
    
    # visualization 
    visualize(model,val_dataset)



if __name__ == '__main__':
    test()