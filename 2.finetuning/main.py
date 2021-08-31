'''
1. 모델 불러오기
'''
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import os
import json
from utils import *



parser = argparse.ArgumentParser(description='FashionMNIST tutorial')
parser.add_argument('--gpu_id', default='1', type=str,
                    help='GPU device')
parser.add_argument('--data_path', default='./data', type=str,
                    help='data path')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--model', default='VGG16', type=str,
                    help='Deep neural network')
parser.add_argument('--data', default='CIFAR10', type=str,
                    help='Dataset')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adagrad', type=str,
                    help='optimizer')
parser.add_argument('--lr', default=0.1e-2, type=float,
                    help='learning rate')
parser.add_argument('--epochs', default=50, type=int,
                    help='train epoch')
parser.add_argument('--weight_decay', default=0.000001, type=float,
                    help='weight_decay')            
args = parser.parse_args()



def train(model, trn_loader, criterion, optimizer, epoch, num_epoch, train_logger):
    model.train()
    train_loss = AverageMeter()
    for i, (data,label) in enumerate(trn_loader):
        data, label = data.cuda(), label.cuda()
        output = model(data)
        loss = criterion(output, label)
        train_loss.update(loss.item()*10000)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss:.4f}'.format(
                epoch, num_epoch, i, len(trn_loader), loss=loss*10000))
    train_logger.write([epoch, train_loss.avg])



def test(model, tst_loader, criterion, epoch, num_epoch, val_logger):
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, (data, label) in enumerate(tst_loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = criterion(output, label)
            val_loss.update(loss.item()*10000)
        print("=================== TEST(Validation) Start ====================")
        print('Epoch : [{0}/{1}]  Test Loss : {loss:.4f}'.format(
                epoch, num_epoch, loss=val_loss.avg))
        print("=================== TEST(Validation) End ======================")
        val_logger.write([epoch, val_loss.avg])



def main():
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # Save configuration
    with open(save_path + '/configuration_' + args.model + '.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # define architecture
    if args.model == 'VGG16': # (1,1000)
        network = models.vgg16_bn(pretrained=True).cuda()
        network = nn.Sequential(network,
                                nn.Linear(1000,10).cuda())
    elif args.model == 'ResNet': # (1,1000)
        network = models.resnet18(pretrained=True).cuda()
        network = nn.Sequential(network,
                                nn.Linear(1000,10).cuda())
    
    network = nn.DataParallel(network).cuda()

    
    # load dataset
    my_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    
    if args.data == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=my_transform,
            )

        val_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            download=True,
            transform=my_transform,
            )


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Train Data Loaded {len(train_dataset)}")
    print(f"Validation Data Loaded {len(val_dataset)}")

    # define criterion
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                         milestones=[int(args.epochs*0.3), int(args.epochs*0.5)],
                                         gamma=0.7)

    # logger
    train_logger = Logger(os.path.join(save_path, 'train_loss_' + args.model + '.log'))
    val_logger = Logger(os.path.join(save_path, 'val_loss_' + args.model + '.log'))


    # training & validation
    for epoch in range(1, args.epochs+1):
        train(network, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
        test(network, val_loader, criterion, epoch, args.epochs, val_logger)
        scheduler.step()
        if epoch%25 == 0 or epoch == args.epochs :
            torch.save(network.state_dict(), '{0}/{1}_{2}.pth'.format(save_path, args.model ,epoch))
    draw_curve(save_path, train_logger, val_logger)
    print("Process complete")


if __name__ == '__main__':
    main()