import torchvision
from torchsummary import summary
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(DEVICE, torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print(DEVICE)
# model = torchvision.models.vgg16(pretrained=True).cuda()
# summary(model,(3,224,224))