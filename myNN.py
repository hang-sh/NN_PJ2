# Train a Network on CIFAR-10 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import random_split
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3,padding=1),
            self.activation,
            nn.BatchNorm2d(64), 
            nn.Conv2d(64, 64, 3,padding=1),
            self.activation,
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(2, 2)
        )
        self.res1 = nn.Conv2d(3, 64, 1, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            self.activation,
            nn.BatchNorm2d(128), 
            nn.Conv2d(128, 128, 3, padding=1),
            self.activation,
            nn.BatchNorm2d(128), 
            nn.MaxPool2d(2, 2)
        )
        self.res2 = nn.Conv2d(64, 128, 1, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            self.activation,
            nn.BatchNorm2d(256), 
            nn.Conv2d(256, 256, 3, padding=1),
            self.activation,
            nn.BatchNorm2d(256), 
            nn.MaxPool2d(2, 2)
        )
        self.res3 = nn.Conv2d(128, 256, 1, stride=2)

        self.dense = nn.Sequential(
            nn.Linear(256*4*4, 768),
            self.activation,
            nn.Linear(768, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        resout1 = self.res1(x)
        x = self.conv1(x)
        x = x + resout1
        # resout2 = self.res2(x)
        x = self.conv2(x)
        # x = x + resout2
        # resout3 = self.res3(x)
        x = self.conv3(x)
        # x = x + resout3
        x = torch.flatten(x, 1) 
        # x = self.gap(x)
        # x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 128

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              # shuffle=True, num_workers=2)
    # 划分训练集和验证集（90%训练，10%验证）
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    print("数据加载成功")
    # 保存最佳模型路径
    dir = './saved_models'
    if not os.path.exists(dir):
        os.makedirs(dir)
    PATH = os.path.join(dir,'cifar_net.pth')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    epochs = 5
    best_acc = 0.0
    # 记录训练和验证历史
    train_loss_hist = []
    val_loss_hist = []
    val_acc_hist = []
    epochs_list = list(range(1, epochs + 1))

    print("开始训练")
    for epoch in range(epochs):  
        net.train()  
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 2000 == 1999:    
                # print(f'Epoch: {epoch + 1}, Iteration: {i + 1:5d}] Loss: {running_loss / 2000:.3f}')
                # running_loss = 0.0
        train_loss = running_loss / len(trainloader)
        train_loss_hist.append(train_loss)
        
        # 验证集评估
        net.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_loss = running_loss / len(valloader)
        val_acc = correct / total
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        print(f'Epoch: {epoch+1} ,Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), PATH)
            print(f"Best model saved at epoch {epoch+1} with val accuracy={val_acc:.4f}")

    print('Finished Training')

    # Test the network 
    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')

    plt.figure(figsize=(10, 8))
    plt.plot(epochs_list, train_loss_hist, label='Train Loss', marker='o')
    plt.plot(epochs_list, val_loss_hist, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot.png')

    plt.figure(figsize=(10, 8))
    plt.plot(epochs_list, val_acc_hist, label='Validation Accuracy', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('accuracy_plot.png')

    