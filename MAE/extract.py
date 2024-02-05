import torch
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

res50 = resnet50(weights=ResNet50_Weights.DEFAULT)
res50_weight = torch.load('resnet50-11ad3fa6.pth')

res50.load_state_dict(res50_weight)

img = np.array(Image.open('scan8/rect_001_max.png'), dtype=np.float32) / 255.

img = torch.Tensor(img).transpose(0, -1).unsqueeze(0)
extract_feature(res50, img)