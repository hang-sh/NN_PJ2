import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_loss_curve(loss_txt_path, save_dir=None, fileneme='loss_curve.png'):
    """
    可视化 loss 曲线，并可选择保存图片
    """
    loss = np.loadtxt(loss_txt_path)

    plt.figure(figsize=(10, 5))
    plt.imshow(loss, aspect='auto', cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel('Batch')
    plt.ylabel('Epoch')
    plt.title('Loss Heatmap')
    plt.grid()
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # y坐标显示整数

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fileneme))
    plt.show()

def plot_grad_curve(grads_txt_path, save_dir=None, fileneme='grad_curve.png'):
    """
    可视化 grads 曲线，并可选择保存图片
    """
    grads = np.loadtxt(grads_txt_path)

    plt.figure(figsize=(10, 5))
    plt.imshow(grads, aspect='auto', cmap='viridis')
    plt.colorbar(label='Grad')
    plt.xlabel('Batch')
    plt.ylabel('Epoch')
    plt.title('Grad Heatmap')
    plt.grid()
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # y坐标显示整数

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fileneme))
    plt.show()


if __name__ == "__main__":

    module_path = os.path.dirname(os.getcwd())
    home_path = module_path

    figures_path = os.path.join(home_path, 'reports', 'figures')
    txt_save_path = os.path.join(home_path, 'reports', 'outputs')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(txt_save_path, exist_ok=True)
    
    # 在对应位置填写.txt文件名和结果文件名
    plot_loss_curve(os.path.join(txt_save_path, 'loss_vgga0.txt'),
                    save_dir=figures_path, fileneme='loss_curve_vgg_a0.png')
    
    plot_grad_curve(os.path.join(txt_save_path, 'grads_vgga0.txt'), 
                    save_dir=figures_path, fileneme='grad_curve_vgg_a0.png')
