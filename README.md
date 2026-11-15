# 基于 ResNet-18 及轻量化改进的交通标志识别

本项目在 GTSRB 德国交通标志数据集上，从随机初始化开始训练 ResNet-18，并以“标准卷积或深度可分离卷积”“不使用或使用 SE 通道注意力”两个因素构成四组对照。实验同时比较参数量、单图乘加次数、分类效果、亮度与模糊扰动下的表现，并提供已裁剪交通标志图像的分类与人工复核接口。

训练指标、混淆矩阵、模型权重、数据划分及源代码哈希保存在运行输出中。四组均已完成 20 轮训练：基线官方测试准确率为 98.2106%；按验证集选择的轻量化应用模型为 `depthwise_se`，测试准确率为 95.3365%，相对基线减少 86.15% 参数和 81.39% MACs。完整分类指标以 `results/summary.json` 和 `results/metrics.csv` 为准，应用与扰动结果以 `results/scene.json` 为准。

## 获取项目

仓库地址：[sichaow5-glitch/neural-network-course](https://github.com/sichaow5-glitch/neural-network-course)。本仓库仅发布交通标志识别大作业，包含源码、汇总指标、逐轮指标和图表。

```sh
git clone https://github.com/sichaow5-glitch/neural-network-course.git
cd neural-network-course
```

以下安装和运行命令均在克隆后的仓库根目录执行。仓库不包含 `checkpoints/`、四组 `results/<variant>_predictions.npz`、`results/provenance/split_manifest.json`，以及原始数据和数组缓存。重新运行数据准备与训练可生成本地数据划分、权重及逐样本预测；已有的汇总记录与图表可直接查看。

## 环境与安装

本次实测环境为 Windows 11、`sam3_course` 环境、Python 3.12.13、NVIDIA GeForce RTX 4060 Laptop GPU。训练采用 CUDA 12.8，软件版本如下。

| 软件 | 实测版本 |
|---|---|
| PyTorch | 2.8.0+cu128 |
| Torchvision | 0.23.0+cu128 |
| NumPy | 1.26.4 |
| Pillow | 12.2.0 |
| Matplotlib | 3.10.8 |

建议使用 Python 3.12 创建独立环境。在项目根目录打开终端，先确认 `python --version`，再执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Linux 或 macOS 的虚拟环境激活命令为 `source .venv/bin/activate`。若只使用 CPU，将上面的 PyTorch 安装命令替换为：

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

之后将训练与场景评价命令中的 `--device cuda` 改为 `--device cpu`。完整训练在 CPU 上会明显变慢。CUDA 运行需要兼容驱动；具体安装版本与支持条件见 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。

## 完整运行顺序

在项目根目录依次运行。首次准备数据需要联网，后续运行会复用经过哈希检查的缓存。从 Git 仓库克隆后，需要完成下面的完整流程，才能在本机重新执行一致性验证；仓库中的历史汇总文件不能替代缺少的本地数据、权重和逐样本预测。

```sh
python data.py
python test_models.py
python train.py --device cuda --resume
python scene.py evaluate --device cuda
python visualize.py
python scene.py classify results/example_input.png --device cpu --output results/example_prediction.json
python verify_results.py
```

`train.py` 默认依次运行 `baseline`、`se`、`depthwise`、`depthwise_se`，每组训练 20 轮，并生成最佳权重及四个逐样本预测文件。首次克隆后即使指定 `--resume`，由于没有本地检查点，仍会从头训练，并更新本地对应的实验结果。`test_models.py` 检查自写基线与 Torchvision ResNet-18 在相同权重下的前向输出精确一致，并检查四组模型的输出形状、有限梯度、参数量和 MACs 统计过程中 BN 状态与训练模式的恢复。

`verify_results.py` 在数据准备、四组训练、场景评价和单图分类全部完成后，重新核对数据划分、实际轨迹交集、预测记录、分类指标、模型权重和源代码哈希，检查记录之间是否一致。核验使用 `data/` 下重新生成的缓存与划分清单，无需仓库中另存 `results/provenance/split_manifest.json`。仅使用最佳权重执行单图分类不需要原始数据缓存。

`--resume` 会跳过配置、源代码、数据划分和最佳权重哈希均匹配的已完成组；中断组从最近完整轮次的 `*_last.pt` 继续，恢复模型、优化器和混合精度缩放器状态。每轮的数据顺序与增强随机流由轮次重新确定。若修改了模型、数据、训练代码或训练配置，应使用新的结果与检查点目录，不能混接旧实验。

例如只验证一轮基线训练流程，可使用独立目录，避免覆盖正式结果：

```sh
python train.py --device cuda --variants baseline --epochs 1 --result-dir smoke-results --checkpoint-dir smoke-checkpoints
```

这条命令只验证运行流程，不构成完整对照实验，也不能直接供要求四组结果齐全的场景评价使用。若复制了已有 GPU 结果后改用 CPU 重新训练，也应指定新的结果与检查点目录；训练精度改变会触发恢复配置检查。

## 数据来源与划分

数据来自 [GTSRB 维护方公布的分发入口](https://github.com/houbensebastian/GermanTrafficSignBenchmarks)，实际文件由官方 ERDA 目录下载：

| 文件 | 用途 | 实测字节数 |
|---|---|---:|
| `GTSRB_Final_Training_Images.zip` | 最终版训练图像及标注 | 276,294,756 |
| `GTSRB_Final_Test_Images.zip` | 官方测试图像 | 88,978,620 |
| `GTSRB_Final_Test_GT.zip` | 官方测试标签 | 99,620 |

`data.py` 检查下载文件的预设 SHA256，记录实际 MD5 与 SHA256，并拒绝包含越界路径或符号链接的压缩包成员。预设 SHA256 取自 [stable-datasets 发布的校验清单](https://galilai-group.github.io/stable-datasets/_modules/stable_datasets/images/gtsrb/)，该清单是第三方公开记录，不能视为数据原作者签发的认证。缓存数组和完整划分清单另有 SHA256 校验。

需要区分两个训练版本：Torchvision 0.23 下载器使用的 `GTSRB-Training_fixed.zip` 实测为 26,640 张，是较早比赛阶段的数据。本项目准备数据时由 39,209 张计数断言发现了这一差异，随后使用 `GTSRB_Final_Training_Images.zip`。不要将旧包的样本量写成最终版样本量，也不要用旧包静默替代本实验的数据。

最终官方训练部分共有 39,209 张图像、43 类。每个训练轨迹通常包含同一个物理交通标志的约 30 帧。验证集按类别内的轨迹划分，轨迹标识采用“类别编号＋文件名下划线前的前缀”，因为不同类别可以使用相同的数字前缀。种子固定为 42，每类取最接近 20% 的完整轨迹作为验证集，并保证训练与验证都包含该类别。

| 数据部分 | 图像数 | 训练来源轨迹数 | 用途 |
|---|---:|---:|---|
| 训练 | 31,380 | 1,046 | 更新参数及估计归一化统计量 |
| 验证 | 7,829 | 261 | 选择检查点、应用模型与置信度阈值 |
| 官方测试 | 12,630 | 不从训练轨迹重新划分 | 最终分类评价与固定扰动评价 |

训练与验证轨迹交集为 0。划分清单保存在 `data/cache_64_seed42_val0p2/split_manifest.json`，其 SHA256 为 `8ba6c9cd6349498794cfdf037a4c615ad37a40a80dc7b5bd8fb0c4b604b5220f`。测试标签不参与模型、轮次或阈值选择。

## 图像处理与训练设置

所有图像转换为 RGB，使用 Pillow 双线性插值将整个输入裁剪图缩放至 64×64，不根据标注框再次裁剪。缓存为 NCHW 排列的 uint8 数组。进入网络前像素除以 255，再按训练部分计算的均值和标准差归一化：

| 通道 | 均值 | 标准差 |
|---|---:|---:|
| R | 0.3410500696697329 | 0.2754734908821420 |
| G | 0.3130976708888573 | 0.26525446761686483 |
| B | 0.3224372077842452 | 0.2713716243945422 |

验证与测试使用相同的统计量，不参与统计量估计。随机数据增强仅用于训练：旋转范围 ±10°、横纵平移范围约 ±6%、缩放范围 0.9–1.1、亮度乘数范围 0.8–1.2。仿射采样使用双线性插值和边界像素填充。不进行水平或垂直翻转，以免改变方向类标志的语义。

| 项目 | 配置 |
|---|---|
| 初始化 | 随机初始化，不加载预训练权重 |
| 四组训练预算 | 每组 20 轮 |
| 训练与评价批大小 | 均为 256 |
| 随机种子 | 训练种子 42，数据划分种子 42 |
| 损失 | 交叉熵，无标签平滑 |
| 优化器 | AdamW，权重衰减 0.0001 |
| 学习率 | 基础值 0.001，前 2 轮预热，随后余弦下降至基础值的 0.02 |
| CUDA 训练 | AMP float16 |
| 验证、测试与场景评价 | FP32 |
| 检查点选择 | 验证准确率最高者；并列保留较早轮次 |
| 执行设置 | CPU 线程数 4，关闭 cuDNN benchmark 与 TF32，启用 cuDNN deterministic |

每组开始时重设相同的训练种子。第 `epoch` 轮的数据排列使用独立 CPU 随机发生器，种子为 `seed * 100000 + epoch`；增强使用独立的计算设备随机发生器，种子为 `seed * 100000 + epoch + 10000`。因此，在相同批大小和环境下，四种架构共享每轮样本顺序及对应的增强随机数，不会因模型初始化消耗随机数的数量不同而改变数据增强。相同种子不意味着不同结构的所有初始权重一一对应，也不保证跨设备、驱动或软件版本逐位一致。

基线设置的项目验收目标为官方测试准确率至少 95%，是否达到由 `baseline.json` 中的 `baseline_target_met` 记录。95% 是本项目预先设定的目标，不是 GTSRB 官方排行榜成绩或特定论文同设置的复现承诺。

## 四组模型与计算量口径

基线保留经典 ResNet-18 的 7×7、步长 2 输入卷积，3×3 最大池化，以及通道数 64、128、256、512 的四个残差阶段。每阶段有两个 BasicBlock，最终全局平均池化和全连接层输出 43 类。

深度可分离变体将残差主分支中的 16 个标准 3×3 卷积，各自替换为“逐通道 3×3＋逐点 1×1”。这两层之间不额外插入 BN 或激活，原有外部 BN/ReLU 位置保持一致。输入卷积和投影捷径仍使用标准卷积。

SE 变体在每个 BasicBlock 的第二个 BN 之后、残差相加之前加入通道注意力：全局平均池化、压缩比 16 的两层无偏置全连接、ReLU 与 Sigmoid 门控。两类改动可独立启用，形成完整的 2×2 对照。

### 实测对照结果

下表从四组训练完成后的 `results/summary.json` 读取；所有模型都训练 20 轮，再按验证准确率选择最佳轮次。参数量与 MACs 由 `profile_model` 对单张 3×64×64 输入实际统计，可通过 `python models.py` 重算。宏平均 F1 按 0–1 的数值尺度表示。

| 变体 | 最佳轮次 | 验证准确率 | 测试准确率 | 测试宏平均 F1 | 参数量 | 单图 MACs |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 20 | 98.7099% | 98.2106% | 0.974155 | 11,198,571 | 148,067,840 |
| `se` | 15 | 98.7099% | 97.5534% | 0.966344 | 11,285,611 | 148,154,880 |
| `depthwise` | 19 | 95.8871% | 94.4576% | 0.917958 | 1,464,235 | 27,474,432 |
| `depthwise_se` | 19 | 97.8669% | 95.3365% | 0.930793 | 1,551,275 | 27,561,472 |

四组在 12,630 张官方测试图上分别判对 12,404、12,321、11,930、12,041 张。基线达到预先设定的 95% 项目验收目标。

| 轻量化变体 | 相对基线参数减少 | 相对基线 MACs 减少 | 相对基线测试准确率下降 |
|---|---:|---:|---:|
| `depthwise` | 86.92% | 81.44% | 3.75297 个百分点 |
| `depthwise_se` | 86.15% | 81.39% | 2.87411 个百分点 |

在本次固定划分、单个种子实验中，标准骨干加入 SE 后测试准确率下降约 0.66 个百分点；深度可分离骨干加入 SE 后则提高约 0.88 个百分点。注意力的作用依赖骨干和训练条件，本次结果不支持“加 SE 必然提高精度”的结论。`depthwise_se` 的验证准确率高于 `depthwise`，因此按照预先制定的验证集选模规则被用于应用接口；其测试准确率仍低于基线，体现出压缩规模与精度之间的取舍。以上差异没有经过多种子显著性检验。

参数量包含学习得到的 BN 参数及分类器偏置，不包含运行均值等缓冲区。MACs 仅累计 Conv2d 和 Linear 的乘加，SE 的两层全连接计入；不计偏置加法、BN、池化、激活、残差相加和 SE 逐元素乘法。一个 MAC 约对应两次浮点运算，但这里不把 MACs 直接写成 FLOPs。MACs 减少也不等于实测延迟按相同比例降低，实际速度还受算子实现、访存和设备利用率影响。

## 场景评价与分类接口

`scene.py evaluate` 对四组最佳检查点使用同一官方测试集，评价以下五种固定条件。扰动施加于缩放后的 RGB 浮点图像，随后执行训练时的归一化。

| 条件 | 操作 |
|---|---|
| 原图 | 不追加扰动 |
| 降低亮度 | RGB 像素乘 0.5，并截断至 [0,1] |
| 提高亮度 | RGB 像素乘 1.5，并截断至 [0,1] |
| 轻度模糊 | 5×5 高斯核，σ=1.0，反射填充 |
| 较强模糊 | 5×5 高斯核，σ=2.0，反射填充 |

每种条件记录准确率、宏平均 F1、各类别召回率平均值、交叉熵及相对原图的准确率下降百分点。应用模型只在 `depthwise` 和 `depthwise_se` 两个轻量化候选中，按原图验证准确率选择；并列时选参数更少者。

应用的置信度阈值也只在验证集确定。预设网格为 `0、0.5、0.6、0.7、0.8、0.85、0.9、0.95、0.97、0.99`：从被接受样本准确率至少 99% 的候选中，选覆盖率最高者，覆盖率并列时选较低阈值；没有合格候选则全部交由人工复核。测试时冻结模型与阈值，并同时报告覆盖率、接受样本数、接受样本正确率和错误数。若没有接受样本，接受样本准确率记为 `null`，不会误写成 100%。

99% 是这份验证数据上的经验筛选条件，不是未来输入的准确率保证、置信度校准结论或道路安全保证。模糊和亮度测试属于固定的合成扰动试验，不能代替真实道路验证。

对自己的已裁剪单个交通标志图片运行：

```sh
python scene.py classify your_sign.png --device cpu --output results/your_prediction.json
```

程序输出类别编号、中英文类别名、最大 Softmax 概率、阈值以及“自动分类”或“需人工复核”。该接口执行德国交通标志的裁剪图分类，不在整张道路图中定位目标，也没有验证中国交通标志类别或自动驾驶控制用途。示例 `example_input.png` 固定使用官方测试集的第一个样本，没有按预测是否成功挑选。

场景评价还实测单张图像的 FP32 前向延迟：预热 10 次、重复 50 次，CUDA 运行时显式同步，输出中位数和第 95 百分位数。计时不包含文件读取、预处理和主机到设备的数据传输；不同硬件上的数值不宜直接比较。

## 文件与输出

| 路径 | 内容 |
|---|---|
| `data.py` | 官方下载、哈希校验、安全解压、轨迹划分与数组缓存 |
| `models.py` | 自写 ResNet-18、两种结构修改及参数/MACs 统计 |
| `test_models.py` | 架构一致性、梯度与统计函数状态恢复检查 |
| `verify_results.py` | 数据划分、预测、指标、模型权重与源代码一致性复核 |
| `train.py` | 四组训练、恢复、验证选模和最终测试 |
| `scene.py` | 固定扰动评价、阈值选择、延迟测量和单图分类 |
| `visualize.py` | 从已记录结果生成曲线、混淆分析及错误样例图 |
| `requirements.txt` | Python 依赖版本 |
| `data/archives/` | 下载的原始压缩包，不提交 Git |
| `data/raw/` | 解压后的原始图片与标签，不提交 Git |
| `data/cache_64_seed42_val0p2/` | uint8 数组、划分清单、训练统计量及哈希，不提交 Git |
| `checkpoints/<variant>.pt` | 验证集选出的最佳权重与模型元数据，不提交 Git |
| `checkpoints/<variant>_last.pt` | 最近完整轮次状态，用于恢复训练，不提交 Git |
| `results/experiment_config.json` | 训练配置、环境、数据来源和每类样本统计 |
| `results/provenance/metadata.json` | 数据来源、哈希、划分统计和归一化参数的保留副本 |
| `results/provenance/split_manifest.json` | 实际训练、验证与测试路径和样本索引的保留副本，不提交 Git |
| `results/provenance/dataset_version_note.json` | 旧训练包版本陷阱及最终版修正的实测记录 |
| `results/<variant>_progress.json` | 逐轮训练与验证进度 |
| `results/<variant>.json` | 一组模型的完整历史、最终评价及检查点哈希 |
| `results/<variant>_predictions.npz` | 四组最佳模型在验证集和测试集上的概率与真实标签，不提交 Git，由训练重建 |
| `results/summary.json` | 已完成各组模型的汇总与代码、划分来源信息 |
| `results/metrics.csv` | 四组参数量、MACs、最佳轮次、精度与宏平均 F1 |
| `results/scene.json` | 五种条件的分类结果、应用阈值策略和延迟原始样本 |
| `results/learning_curves.png` | 训练与验证曲线 |
| `results/metrics_comparison.png` | 分类效果及模型规模对比 |
| `results/confusion_matrix.png` | 应用模型的测试混淆矩阵 |
| `results/confusion_pairs.png` | 主要混淆类别对 |
| `results/test_errors.png` | 测试错误样例 |
| `results/robustness_curves.png` | 各模型在固定扰动下的表现 |
| `results/plot_data.json` | 图表使用的数值与来源记录 |
| `results/example_input.png` | 固定测试样例，供单图命令演示 |
| `results/example_prediction.json` | 由单图分类命令生成的单图输出 |
| `results/verification.json` | 数据、代码、检查点、预测与汇总指标的一致性核验结果 |

Git 仓库保存源码、汇总与逐轮指标、场景记录及图表，不上传原始数据、数组缓存、`checkpoints/`、四个 `results/<variant>_predictions.npz` 和 `results/provenance/split_manifest.json`。文件表同时列出本地完整运行后可生成的主要文件，不能理解为这些文件均已提交至仓库。

已交付的 ZIP 包包含四组最佳权重和逐样本预测文件，解压后可配合包内结果执行单图分类；重新进行完整场景评价和一致性核验仍需先运行 `python data.py` 准备本地数据。ZIP 未提供公共下载链接。最佳权重用于推理，恢复中断训练还需要本地对应的 `*_last.pt` 优化器状态。修改模型或权重后应重新生成匹配的场景记录，程序会检查已记录的权重哈希。

## 结果解释边界

本次是固定划分、单个训练种子下的四组结构对照。宏平均 F1 平等考虑各类别，准确率则按样本计数，二者应结合类别支持数及混淆矩阵解释。单次精度差异不能视为多种子统计显著性；没有额外实验时，不能把结构修改描述为普遍优于基线。参数量、MACs、设备上的前向延迟和扰动下的准确率分别反映不同方面，需要同时报告。

## 主要来源

- GTSRB 数据维护仓库：[German Traffic Sign Benchmarks](https://github.com/houbensebastian/GermanTrafficSignBenchmarks)。数据论文：Stallkamp 等，*The German Traffic Sign Recognition Benchmark: A multi-class classification competition*，IJCNN 2011；以及 *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*，Neural Networks 2012，[DOI](https://doi.org/10.1016/j.neunet.2012.02.016)。
- ResNet 原论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。基线对照实现：[Torchvision 0.23 ResNet 源码](https://docs.pytorch.org/vision/0.23/_modules/torchvision/models/resnet.html)。
- 深度可分离卷积参考：[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)。
- SE 通道注意力参考：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)。
- 下载校验清单：[stable-datasets GTSRB 源码](https://galilai-group.github.io/stable-datasets/_modules/stable_datasets/images/gtsrb/)。

数据与第三方依赖的使用条件以各自来源说明为准。本项目不额外承诺第三方数据或依赖的再分发许可。
