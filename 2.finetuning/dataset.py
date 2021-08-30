'''
1. 데이터셋을 어떻게 불러오고
2. Transform을 어떻게 하고
3. mean, std로 정규화하는 방법
'''
import torch
import torchvision
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Fine tuning tutorial')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='GPU device')
parser.add_argument('--dataset', default='CIFAR10', type=str,
                    help='Dataset')                         
args = parser.parse_args()



my_transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])



if args.dataset == 'CIFAR10':
    dataset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=my_transform)

dataloader = DataLoader(dataset,shuffle=False,batch_size=32)


def visualize(dataset):
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
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title('Target :'+label_dic[label])
        plt.axis("off")
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    visualize(dataset)