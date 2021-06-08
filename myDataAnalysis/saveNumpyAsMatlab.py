import numpy as np
import scipy.io
import torch

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

arr = np.arange(3*3*3)
arr = arr.reshape((3, 3, 3))  # 2d array of 3x3
print(arr)

a_tensor = torch.rand([5, 5, 5], dtype=torch.double) * 2  # 产生[0,1]均匀分布的数据
a_np = a_tensor.numpy()
print(a_np)

x = np.array([1, 2 , 3])
print(x)


scipy.io.savemat('./myDemoData/arrdata.mat', mdict={'train': {'arr': arr, 'a_np': a_np, 'x': x}})



print('frames_finalpass/TEST/A/0000/left/0006.png'.replace('/', '_'))

IMAGENET_MEAN = torch.Tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.Tensor([0.229, 0.224, 0.225])

print(IMAGENET_MEAN)
print(IMAGENET_STD)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))