***
# Torch Tutorial - Quick start

- 토치 튜토리얼 [링크](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)의 코드를 참고함
- jupyter에서 활용 가능하게 만들어진 코드를 객체 지향적으로 수정
- utils, main, model, dataset 모듈을 추가함
- ~~사실 FashionMNIST는 pytorch내부에 데이터가 있어서 사용자정의 dataset class가 필요없습니다~~...

*** 
# Prerequisites
- Python 3.6
- Pytorch 3.9x
- CUDA 11.1 (of cousrse GPU)
- torchvision
- torchsummary (pip install torchsummary)
- torchmetrics (pip install torchmetrics)
> dataset : Fashion-MNIST (is already built in pytorch)

***

# Usage
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
- Accuracy : [0.7614 ± 0.056]
***

# Qualitative Results
## (1) Train-val curve
[](https://github.com/HongSungRae/Pytorch_tutorial/blob/main/1.quickstart/exp/loss_curve.png?raw=true)

## (2) Random test plot
[](https://github.com/HongSungRae/Pytorch_tutorial/blob/main/1.quickstart/exp/pred_1.png?raw=true)