***
# Torch Tutorial - Fine tuning tutorial
- 토치 튜토리얼 [링크](https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html)의 코드를 참고함
- 객체지향적인 코드

## Journey
1. 파이토치 내부에 구현된 네트워크를 Freeze후 학습에 사용된 데이터셋으로 성능테스트
2. 파이토치 내부에 구현된 네트워크를 다른 데이터셋으로 finetuning 후 성능테스트
3. 저장과 불러오기

## Network
</br>
1. VGG16

![](https://github.com/HongSungRae/Pytorch_tutorial/blob/main/2.finetuning/exp/vggnet.png?raw=true)
![](https://github.com/HongSungRae/Pytorch_tutorial/blob/main/2.finetuning/exp/vggnet2.png?raw=true)
</br>
2. ResNet

![](https://github.com/HongSungRae/Pytorch_tutorial/blob/main/2.finetuning/exp/resnet.png?raw=true)
![](https://github.com/HongSungRae/Pytorch_tutorial/blob/main/2.finetuning/exp/resnet2.png?raw=true)

## Augmentation

*** 
# Prerequisites
- Python 3.6
- Pytorch 3.9x
- CUDA 11.1 (of cousrse GPU)
- torchvision
- torchsummary (pip install torchsummary)
- torchmetrics (pip install torchmetrics)
> CIFAR10

***
# Usage
- It's OK using _Default_ setting if you don't mind.
## Train
```
>python main.py --gup_id <YOUR GPU ID> --data_path <WHERE IS DATA?> --save_path <'./exp'> --batch_size <bs> --optim <Optimizer> --lr <LR> --epochs <EPOCHS> --weight_decay <Decay rate>
```

## Test
- Output : Accuracy, Accuracy std, 9 random predictions
```
>python evaluate.py --gup_id <YOUR GPU ID> --data_path <WHERE IS DATA?> --save_path <'./exp'> --batch_size <bs>
```

***
# Qualitative Results
- 
***

# Qualitative Results
## (1) Train-val curve
![]()

## (2) Random test plot
- Do you think the model predicts well?
![]()