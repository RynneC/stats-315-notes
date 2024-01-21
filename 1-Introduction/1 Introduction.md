# 1 Introduction

本章reading: 

(1) Deep Learning; (2) Deep Learning with Python; (3) Dive into Deep Learning; 的 Chapter 1.

## 1.1 AI, ML, Dl 及历史

<img src="..\NoteAssets\image-20240121132932701.png" alt="image-20240121132932701" style="zoom:67%;" />

### 1.1.1 AI

**AI 做的事情是 automate intellectual tasks normally performed by humans. ML 则是 AI 的一个分支，而 DL 又是 ML的一个分支. **

20 世纪 50 年代到 80 年代末，AI 的主流是 **symbolic AI (符号主义人工智能)**，即程序员为程序编写足够多的规则，通过规则的数量来模拟人. 这一方法的顶峰是 20 世纪 80 年代的**专家系统 (expert system).**

但这种方法只对于逻辑明确的问题有用. 因而就出现了新的方法叫做 **Machine Learning**.

### 1.1.2 ML

一个 machine learning system 是被 trained 出而不是被 explicitly programmed 出来的.

<img src="..\NoteAssets\image-20240121155034565.png" alt="image-20240121155034565" style="zoom: 67%;" />

ML 和数理统计关系很大，但是并不同. ML 用于处理复杂的、高维度的大型 dataset. 对如这种数据。经典的统计分析比如贝叶斯分析是不可能的. 而 ML 尤其是 DL， 用相对少的数学理论，以工程为导向进行处理.

我们需要以下这几个组件来进行 ML：

1. **Input Data Points (数据). **比如 speech recognition 的 data points 为声音，image tagging 的 data points 为 picture.
2. **Model (模型)**: to transform the data. 
3. **Objective Function (目标函数) 或叫 Loss Function:** 计算 algorithm 的 output 与expected output 的差距，检测 model 的有效性.
4. **Optimization Algorithm (优化算法)**: 接收到 Loss Function 的结果之后，调成参数，从而 optimize Loss Function, 

#### 1.1.2.1 Data 数据

每个 Dataset 由一个个 **sample(样本)** 组成，大多时候遵循 **i,i,d (independently and identically distributed，独立同分布).** sample 也叫 **data point**.

每个 sample 由一组 **features (特征)**，或叫 **covariates (协变量)** 组成. 机器学习模型会根据这些 features 进行预测.  在 **supervised learning (监督学习)** 问题中，要预测的是一个特殊的 feature，被称为 **label (标签)** 或 **target (目标)**.

比如处理图像数据时，每一张单独的照片即为一个 sample，它的 features 由每个像素数值的有序列表示. 比如，200 × 200彩色照片由200 × 200 × 3 = 120000个数值组成，“3”对应于每个空间位置的红、绿、蓝强度.

当每个 sample 的 feature 类别的数量都是相同的时候，其eigenvector 是 fixed-length 的，这个长度被称为数据的 dimensionality (维数).

fixed-length 的 eigenvector 是很适合学习的，但是并不是所有的数据都可以用 fixed-length 的 vector 表示. 比如来自互联网的分辨率和形状不同的图像，以及文本数据.

**与传统 ML 方法相比，DL 的一个主要优势是可以处理不同长度的数据.**

#### 1.1.2.2 Model 模型





#### 1.1.2.3 Loss Function 目标函数

当任务在试图预测数值时，最常见的损失函数是 **squared error**，即预测值与实际值之差的平方.

当试图解决分类问题时，最常见的目标函数是 **error rate**，即预测与实际情况不符的样本比例.

有些 Loss Func（如squared error）很容易被优化，有些目标（如 error rate）由于 non-differentiability 或其他复杂性难以直接优化. 这种时候通常会优化 **a surrogate objective (代替目标)**.

通常，损失函数是根据模型 parameters 定义的，并取决于dataset. 在一个数据集上，我们可以通过最小化总损失来
学习模型 parameters 的最佳值.

。该数据集由一些为训练而收集的样本组成，称为训练数据集（training dataset，或
称为训练集（training set））。然而，在训练数据上表现良好的模型，并不一定在“新数据集”上有同样的性
能，这里的“新数据集”通常称为测试数据集（test dataset，或称为测试集（test set））。
Dataset 通常可以分成两部分：训练数据集用于拟合模型参数，测试数据集用于评估拟合的模
型。然后我们观察模型在这两部分数据集的性能。“一个模型在训练数据集上的性能”可以被想象成“一个学
生在模拟考试中的分数”。这个分数用来为一些真正的期末考试做参考，即使成绩令人鼓舞，也不能保证期
末考试成功。换言之，测试性能可能会显著偏离训练性能。当一个模型在训练集上表现良好，但不能推广到
测试集时，这个模型被称为过拟合（overfitting）的。就像在现实生活中，尽管模拟考试考得很好，真正的考
试不一定百发百中。

#### 1.1.2.4 Optimization Algorithm 优化算法

优化算法搜索出 loss func 的最佳 parameters，从而 minimizing loss func.

DL 中大部分流行的 Optim Algo 都基于 Gradient Descent approach. 在 Gradient Descent Approach 在每个步骤都会检查每个 parameter，看看对于某一个 parameter 如果仅改动它的话 loss 会朝哪个方向移动，然后在减少 loss 的方向上进行优化.



### 

