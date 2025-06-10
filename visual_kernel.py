from myNN import Net
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

PATH = 'saved_models\cifar_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('./report'):
    os.makedirs('./report')

net = Net()
net.load_state_dict(torch.load(PATH, map_location=device), strict=False)

kernels = net.conv1[0].weight.data.cpu()  # 形状 [64, 3, 3, 3]
num_kernels = 12 # 只显示前12个卷积核

plt.figure(figsize=(12, 6))
for i in range(num_kernels):
    kernel = kernels[i]  # 形状 [3, 3, 3]
    min_val = kernel.min()
    max_val = kernel.max()
    kernel = (kernel - min_val) / (max_val - min_val) # 归一化到0-1
    kernel = np.transpose(kernel.numpy(), (1, 2, 0))   
    plt.subplot(3, 4, i+1)
    plt.imshow(kernel)
    plt.axis('off')
    plt.title(f'Kernel {i+1}')

plt.suptitle('First Conv Layer Kernels')
plt.savefig('report/first_conv_kernels.png')
plt.show()
