from myNN import Net
import loss_landscapes 
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

if not os.path.exists('./report'):
    os.makedirs('./report')
    
# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

PATH = 'saved_models\cifar_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = Net()
net.load_state_dict(torch.load(PATH, map_location=device), strict=False)
net.to(device)
net.eval()  # 进入评估模式

# 获取一个 batch 的测试数据用于计算损失
inputs, labels = next(iter(testloader))
inputs, labels = inputs.to(device), labels.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑的交叉熵损失

# 定义一个度量函数，该函数接受模型并返回损失值
def metric(model:Net):
    with torch.no_grad():
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
    return loss.item()

# 计算损失景观
landscape = loss_landscapes.random_plane(
    model=net,
    metric=metric,  
    steps=30,
    distance=1.0
)

steps = landscape.shape[0]
x = np.linspace(-1, 1, steps)
y = np.linspace(-1, 1, steps)
X, Y = np.meshgrid(x, y)
Z = landscape

# 绘制损失景观
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Direction 1')
ax.set_ylabel('Direction 2')
ax.set_zlabel('Loss')
plt.title('Loss Landscape')
plt.show()
plt.savefig('report/loss_landscape.png')