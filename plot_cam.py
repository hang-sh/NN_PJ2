from myNN import Net
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

PATH = 'saved_models\cifar_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net()
model.load_state_dict(torch.load(PATH, map_location=device), strict=False)
model.to(device)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 4
trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = next(dataiter)

img = images[0]
img_tensor = img.unsqueeze(0).to(device)  # [1, 3, 32, 32]
rgb_img = img.cpu().numpy()  
rgb_img = np.transpose(rgb_img, (1, 2, 0))  # 转换为 HWC 格式
rgb_img = rgb_img * 0.5 + 0.5  # 反归一化

# 最后一层卷积层
target_layers = [model.conv3[3]]

# 预测类别
model.eval()
with torch.no_grad():
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

targets = [ClassifierOutputTarget(pred_class)]   

# 计算CAM
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

# 叠加可视化
cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 并排显示
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(rgb_img)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(cam_img)
axs[1].set_title(f'GradCAM: {classes[pred_class]}')
axs[1].axis('off')

plt.tight_layout()
plt.show()
# plt.savefig('report/cam_result.png')  # 保存结果