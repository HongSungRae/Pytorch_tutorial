from numpy import datetime_data
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import os
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from utils import * #visualize
import torch.nn as nn
from torchmetrics.functional import accuracy



parser = argparse.ArgumentParser('Test model')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='GPU device')
parser.add_argument('--data_path', default='./data', type=str,
                    help='data path')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--model', default='VGG16', type=str,
                    help='Model that you want to test')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def main():
    # 테스트 데이터셋
    my_transform = torchvision.transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    
    test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            download=True,
            transform=my_transform,
            )
    
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)

    # 모델 불러오기
    if args.model == 'VGG16': # (1,1000)
        network = torchvision.models.vgg16_bn(pretrained=False).cuda()
        network = nn.Sequential(network,
                                nn.Linear(1000,10).cuda())
        network.load_state_dict(torch.load(args.save_path+'/VGG16_2.pth'),strict=False)

    elif args.model == 'ResNet': # (1,1000)
        network = torchvision.models.resnet18(pretrained=True).cuda()
        network = nn.Sequential(network,
                                nn.Linear(1000,10).cuda())
        network.load_state_dict(torch.load(args.save_path+'/ResNet_2.pth'),strict=False)
    network.eval()
    network = nn.DataParallel(network).cuda()

    # # Accuracy
    # acc = AverageMeter()
    # with torch.no_grad():
    #     for i,(data,label) in enumerate(test_loader):
    #         data, label = data.cuda(), label.cuda()
    #         pred = network(data)
    #         acc.update(accuracy(pred,label).item(),n=data.shape[0])
    #         if (i+1)%10==0:
    #             print(f'Now testing on : [{i+1}/{len(test_loader)}] | Acc : [{acc.avg}]')
    # print(f'Avg Accuracy : [{acc.avg}]')
    # print(f'Accuracy std : [{acc.std}]')

    # 시각화
    label_dic = {
                0 : 'airplane',
                1 : 'automobile',
                2 : 'bird',
                3 : 'cat',
                4 : 'deer',
                5 : 'dog',
                6 : 'frog',
                7 : 'horse',
                8 : 'ship',
                9 : 'truck'
                }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
        img, label = test_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        pred = torch.argmax(network(torch.unsqueeze(img,0)),dim=-1).item()
        plt.title('Prediction :'+label_dic[pred])
        #plt.title(label_dic[label])
        plt.axis("off")
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()