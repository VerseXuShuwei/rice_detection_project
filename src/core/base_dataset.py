"""
抽象数据加载器基类：BaseDataset 或 BaseDataModule

目的： 统一数据加载、预处理和增强的接口。

内容：

抽象方法： __init__ (定义数据路径/配置)、__len__、__getitem__。

可选抽象方法： setup() (用于数据拆分/下载)、train_dataloader()、val_dataloader() 等（如果是使用 PyTorch Lightning 风格的 DataModule）。

用途： 确保所有数据集（如 ImageNet, COCO, 您的私有数据）都能以统一的方式被训练脚本调用。

避免过度抽象： 不需要抽象每一种数据增强操作，只抽象数据集的加载和访问接口即可。

"""