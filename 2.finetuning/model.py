'''
학습된 모델을 어떻게 가져오는지 보여주는 예시
더 많은 모델은 https://pytorch.org/vision/stable/models.html 를 참조

pretrained : 사전학습여부
progress : 다운로드 과정을 출력할 것인지? default = True
'''
import torchvision
from torchsummary import summary
import argparse



parser = argparse.ArgumentParser(description='Fine tuning tutorial')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='GPU device')
parser.add_argument('--model', default='VGG16', type=str,
                    help='Deep Neural Networks')                         
args = parser.parse_args()


if args.model == 'VGG16':
    model = torchvision.models.vgg16(pretrained=True, progress=True)
    summary(model,(3,224,224),batch_size=1,device='cpu')
elif args.model == "VGG16_bn":
    model = torchvision.models.vgg16_bn(pretrained=True, progress=True)
    summary(model,(3,224,224),batch_size=1,device='cpu')
elif args.model == "ResNet18":
    model = torchvision.models.vgg16_bn(pretrained=True, progress=True)
    summary(model,(3,224,224),batch_size=1,device='cpu')