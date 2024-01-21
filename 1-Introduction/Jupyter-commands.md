1. 初始化 shell 以能运行 conda

```shell
~/miniconda3/bin/conda init
```

2. 创建新环境

```shell
conda create --name d2l python=3.9 -y
```

3. 激活环境

```shell
conda activate d2l
```

4. 安装pytorch

```shell
pip install torch==1.12.0
pip install torchvision==0.13.0
```

5. 安装 d2l 包

```shell
pip install d2l==0.17.6
```

6. 下载 D2l Notebook

```shell
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd pytorch
```

```shell
sudo apt install unzip
```

7. 打开 notebook

```shell
jupyter notebook
```

8. 退出环境

```shell
conda deactivate
```

