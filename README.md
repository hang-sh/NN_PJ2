# Project-2 of “Neural Network and Deep Learning”

本项目为课程神经网络和深度学习PJ2.



## 数据集下载

[数据集下载](https://pan.baidu.com/s/1zWjv2NSyJHpbjGl9KtLQHw?pwd=kw6v)

下载完成后，请将数据文件放入`data/` 文件夹中，运行文件 `loaders.py` 即可完成数据集下载和查看。



## Task 1: Train a Network on CIFAR-10

训练并测试自定义网络

```bash
python myNN.py
```

训练完成的模型会自动保存至`saved_models/` 文件夹下，并自动绘制训练历史保存至当前文件夹下。



可视化

**Loss Landscape**

```bash
python plot_loss_landscape.py
```

**卷积核可视化**

```bash
python visual_kernel.py
```

**CAM**

```bash
python plot_cam.py
```



## Task 2:  Batch Normalization

**VGG-A with and without BN**

打开 `VGG_Loss_Landscape.py` 文件，在`if __name__ == '__main__':` 下代码中选择需要训练的模型，同时可自定义模型文件名和实验名，并运行文件：

```bash
python VGG_Loss_Landscape.py
```

训练完成后模型和历史信息将保存至项目根目录下的`reports` 文件夹。



loss 和 grad 可视化

打开 `plot_loss_grad.py` 文件，填写需要绘制的`.txt` 文件名，运行文件：

```bash
python plot_loss_grad.py
```

结果将保存至项目根目录下的`reports/figures` 文件夹。



**Loss Landscape**

打开 `VGG_Loss_Landscape.py` 文件，在`if __name__ == '__main__':` 下代码中将第一部分注释掉，运行第二部分代码：

```python
python VGG_Loss_Landscape.py
```

结果将保存至项目根目录下的`reports` 文件夹。