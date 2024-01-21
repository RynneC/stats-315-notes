# 1 Introduction

本章reading: 

(1) Deep Learning; (2) Deep Learning with Python; (3) Dive into Deep Learning; 的 Chapter 1.

## 1.1 AI, ML, Dl 及历史

<img src="C:\Users\19680\Documents\GitHub\23W-STATS-315\NoteAssets\image-20240121132932701.png" alt="image-20240121132932701" style="zoom:67%;" />

### 1.1.1 AI

**AI 做的事情是 automate intellectual tasks normally performed by humans. ML 则是 AI 的一个分支，而 DL 又是 ML的一个分支. **

20 世纪 50 年代到 80 年代末，AI 的主流是 **symbolic AI (符号主义人工智能)**，即程序员为程序编写足够多的规则，通过规则的数量来模拟人. 这一方法的顶峰是 20 世纪 80 年代的**专家系统 (expert system).**

但这种方法只对于逻辑明确的问题有用. 因而就出现了新的方法叫做 **Machine Learning**.

### 1.1.2 ML

一个 machine learning system 是被 trained 出而不是被 explicitly programmed 出来的.

<img src="NoteAssets\image-20240121133356444.png" alt="image-20240121133356444" style="zoom:67%;" />

ML 和数理统计关系很大，但是并不同. ML 用于处理复杂的、高维度的大型 dataset. 对如这种数据。经典的统计分析比如贝叶斯分析是不可能的. 而 ML 尤其是 DL， 用相对少的数学理论，以工程为导向进行处理.

我们需要以下三个要素来进行机器学习。

1. **Input Data Points. **比如 speech recognition 的 data points 为声音，image tagging 的 data points 为 picture.
2. **Examples of Expected output**. 比如 image tagging 的expected outputs 是“狗”“猫”之类的标签。
3. **检测 algorithm 好坏的办法.** 计算 algorithm 的 output 与expected output 的差距. 而这个检测的结果又是一个 **feedback signal**，可以用来 adjust algorithm. 因而这样 检测而后 feedback 而后 adjust 的方式正是我们说的 learning.

输入数据变换为有意义的输出，这是一个从已知的输入和输出示例中进行
“学习”的过程。因此，机器学习和深度学习的核心问题在于有意义地变换数据，换句话说，在
于学习输入数据的有用表示（representation）——这种表示可以让数据更接近预期输出。在进一
步讨论之前，我们需要先回答一个问题：什么是表示？这一概念的核心在于以一种不同的方式
来查看数据（即表征数据或将数据编码）。