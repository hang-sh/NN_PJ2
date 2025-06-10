import matplotlib as mpl
mpl.use('Agg') # 将 matplotlib 的后端设置为 'Agg'，即非交互式后端
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm 
from data.loaders import get_cifar_loader

import torchvision
import os
import matplotlib.ticker as ticker


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ## Constants (parameters) initialization
# device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path

figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
loss_save_path = os.path.join(home_path, 'reports', 'outputs')
grad_save_path = os.path.join(home_path, 'reports', 'outputs')

if not os.path.exists(figures_path):
    os.makedirs(figures_path)

if not os.path.exists(models_path):
    os.makedirs(models_path)

if not os.path.exists(loss_save_path):
    os.makedirs(loss_save_path)

# Make sure you are using the right device.
# device_id = device_id
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.get_device_name(3))


# Initialize your data loader
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True, n_items=3)
val_loader = get_cifar_loader(train=False, n_items=3)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(os.path.join(figures_path, 'sample.png'))
    # plt.show()

def show_sample():
    # Show one sample from the train_loader
    display.clear_output(wait=True)
    print('Sample from train_loader:')
    for X,y in train_loader:
        ## --------------------
        # Add code as needed
        n = 4
        imshow(torchvision.utils.make_grid(X[:n]))
        print(' '.join(f'{classes[y[j]]:5s}' for j in range(n)))
        ## --------------------
        break

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, dataloder, device=device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloder:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc
    

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, exp_name='vgg_a'):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 记录loss和grad的均值
            loss_list.append(loss.item())
            grad_batch = np.array(model.classifier[4].weight.grad.clone().cpu())
            grad_mean = grad_batch.mean()
            grad.append(grad_mean)
            learning_curve[epoch] += loss.item()
            ## --------------------
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        _ , axes = plt.subplots(1, 2, figsize=(15, 3))
        # 设置横轴只显示整数
        axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve, label='Train Loss')

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        train_accuracy = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_accuracy
        val_accuracy_curve[epoch] = val_accuracy
        print(f"Epoch {epoch+1}, Train Loss: {learning_curve[epoch]:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {max_val_accuracy_epoch+1} with val accuracy={max_val_accuracy:.4f}")
        
        axes[1].plot(train_accuracy_curve, label='Train Accuracy')
        axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
        
        if scheduler is not None:
            scheduler.step()
        ## --------------------

    axes[0].set_xlabel('Epochs')
    axes[1].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Accuracy')
    axes[0].set_title('Training Loss Curve')
    axes[1].set_title('Training and Validation Accuracy Curve')
    axes[0].legend('Train Loss')
    axes[1].legend(['Train Accuracy', 'Validation Accuracy'])
    
    plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(figures_path, f'train_hist_{exp_name}.png'))

    return losses_list, grads

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve_list, max_curve_list, label_list=['Standard VGG', 'Standard VGG + BatchNorm'], colors=['green', 'red']):
    epo = np.arange(len(min_curve_list[0]))        
    plt.figure(figsize=(10, 5))
    for i in range(len(min_curve_list)):
        plt.plot(epo, min_curve_list[i], color=colors[i])
        plt.plot(epo, max_curve_list[i], color=colors[i])
        plt.fill_between(epo, min_curve_list[i], max_curve_list[i], label=label_list[i], alpha=0.3, color=colors[i])
    
    plt.xlabel('Steps')
    plt.ylabel('Loss Landscape')
    plt.title('Loss Landscape')
    plt.grid()
    plt.legend() 
    # 设置横轴只显示整数
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'))
    # plt.show()

if __name__ == '__main__':

    set_random_seeds(seed_value=2020, device=device)

    # show_sample()
    
# ========== 第一部分 ==========
    # Train your model
    # feel free to modify
    epo = 25

    model = VGG_A()
    # model = VGG_A_BatchNorm()
    
    lr = 0.001 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) 
    # 可自定义模型文件名和实验名
    loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, scheduler=scheduler, 
                        best_model_path = os.path.join(models_path, 'vgg_a.pth'), 
                        exp_name = 'vgg_a')
    np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')
# ========== 第一部分 ==========

# ========== 第二部分 ==========

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
## --------------------
# Add your code

    epo = 20
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑

    model_list = ['VGG_A', 'VGG_A_BatchNorm']
    lr_list = [1e-3, 1e-4, 5e-4, 5e-5]
    min_curve_list = []
    max_curve_list = []

    for net in model_list:
        min_curve = []
        max_curve = []

        all_losses = []
        for lr in lr_list:
            print(f'Training {net} with learning rate {lr}:')

            if net == 'VGG_A':
                model = VGG_A()
            elif net == 'VGG_A_BatchNorm':
                model = VGG_A_BatchNorm()

            optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) 
            loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=os.path.join(models_path, f'{net}_lr_{lr}.pth'),scheduler=scheduler,exp_name=f'{net}_lr_{lr}')
            loss = np.array(loss).reshape(-1).squeeze()
            all_losses.append(loss)

        min_curve = np.amin(all_losses, axis=0)
        max_curve = np.amax(all_losses, axis=0)
    
        min_curve_list.append(list(min_curve))
        max_curve_list.append(list(max_curve))
        
        
    plot_loss_landscape(min_curve_list, max_curve_list)

# ========== 第二部分 ==========
