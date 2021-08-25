import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from model import NeuralNetwork # 내가 정의한 네트워크
import argparse
import os
import json
from torch.optim import lr_scheduler
from utils import Logger, AverageMeter, draw_curve


parser = argparse.ArgumentParser(description='FashionMNIST tutorial')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='GPU device')
parser.add_argument('--data_path', default='./data', type=str,
                    help='data path')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adagrad', type=str,
                    help='optimizer')
parser.add_argument('--lr', default=0.1e-2, type=float,
                    help='learning rate')
parser.add_argument('--epochs', default=30, type=int,
                    help='train epoch')
parser.add_argument('--weight_decay', default=0.000001, type=float,
                    help='weight_decay')
# parser.add_argument('--test_val_split', default=0.7, type=float,
#                     help='test set : val set = test_val_split : 1-test_val_split')                
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
        with open(save_path + '/configuration.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # define architecture
    network = NeuralNetwork().cuda()
    network = nn.DataParallel(network).cuda()

    # load dataset
    train_dataset = datasets.FashionMNIST(
        root=args.data_path,
        train=True,
        download=True,
        transform=ToTensor(),
        )

    val_dataset = datasets.FashionMNIST(
        root=args.data_path,
        train=False,
        download=True,
        transform=ToTensor(),
        )


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
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
    train_logger = Logger(os.path.join(save_path, 'train_loss.log'))
    val_logger = Logger(os.path.join(save_path, 'val_loss.log'))

    # training & validation
    for epoch in range(1, args.epochs+1):
        train(network, train_loader, criterion ,optimizer, epoch, args.epochs, train_logger)
        test(network, val_loader, criterion, epoch, args.epochs, val_logger)
        scheduler.step()
        if epoch%20 == 0 or epoch == args.epochs :
            torch.save(network.state_dict(), '{0}/{1}_{2}.pth'.format(save_path, 'my_network' ,epoch))
    draw_curve(save_path, train_logger, val_logger)
    print("Process complete")


if __name__ == '__main__':
    main()