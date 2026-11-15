# 来源与第三方说明

模型结构依据 ResNet、MobileNets 和 Squeeze-and-Excitation 的公开论文实现。基线模块组织和初始化参考 Torchvision 0.23 的 ResNet，实现正确性通过加载同一 state_dict 与 Torchvision 的输出逐值比较核验。Torchvision 的 BSD 3-Clause 许可见本目录 TORCHVISION_LICENSE。

- ResNet 原论文：https://arxiv.org/abs/1512.03385
- 深度可分离卷积参考：https://arxiv.org/abs/1704.04861
- SE 原论文：https://arxiv.org/abs/1709.01507
- Torchvision ResNet 源码：https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/resnet.py
- Torchvision 许可：https://github.com/pytorch/vision/blob/v0.23.0/LICENSE

GTSRB 图像属于原数据集，本项目未将完整数据集上传到代码仓库。数据获取自官方维护者链接的 ERDA 归档，下载程序保留来源、摘要和原始标注；数据的使用应遵循原发布条件。结果图中的交通标志图像是该数据集的实验样例，不是新创作图像。类别中文名为项目中的解释性译名，不构成交通法规说明。

- GTSRB 官方维护页：https://github.com/houbensebastian/GermanTrafficSignBenchmarks
- 官方归档：https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
- 数据论文：Stallkamp, Schlipsing, Salmen, Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks, 2012. https://doi.org/10.1016/j.neunet.2012.02.016
- 预设 SHA256 的第三方来源：https://galilai-group.github.io/stable-datasets/_modules/stable_datasets/images/gtsrb/

预设 SHA256 来自第三方发布清单；它用于检测下载不一致，不被表述为数据作者签发的认证。下载地址来自官方归档。原始第三方软件和数据的许可不会因本项目的封装而改变。
