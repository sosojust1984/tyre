# 一、项目简介

轮胎检测是主机厂整车设计中必不可少的重要环节，虽然出厂阶段轮胎厂商已经按照主机厂提出的标准进行了相应的检测，但为了保证实车的安全性，主机厂还需要进行多路段的实车轮胎检测。具体过程如下：

**图一、检测路段**

![](https://ai-studio-static-online.cdn.bcebos.com/df5535cabfa54606841c721f9e93aff2d5d4b632bc9e4af89a86453cca9c4763)

**图二、实验过程**

![](https://ai-studio-static-online.cdn.bcebos.com/469eda0a6aa1403189d48dabdc246051e6c6cae745494f50a64a07405771dd7b)

**图三、检测结果**

![](https://ai-studio-static-online.cdn.bcebos.com/986fe20b871045fda96d0d8181083fada3ed067381e7450781e6797fbdc13f4d)



通过上面三张图片展示可以清楚的看到，由于主机厂主要从事整车的设计无法配套轮胎厂商的大型检测设备，所以用简单的测量仪器进行试验。试验数据手写记录在笔记本上，为了保证数据的准确性需要进行多次测量取平均值，根据实际标准要求每种车型需要两辆车八组轮胎测试数据，最终产生数据上万条。且对于大型主机厂来说每年有大量的设计车型需求，所以将产生大量的手写数据表格。将手写数据表格输入到电子表格是一个重复繁杂的工作。
为了提高试验效率考虑用百度的PaddleOCR进行识别转化，为主机厂减负提高效率。





# 二、配置环境


```python
# clone PaddleOCR代码
! git  clone https://github.com/PaddlePaddle/PaddleOCR

# 安装依赖包
! pip install -U pip
! pip install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
! pip install -r PaddleOCR/requirements.txt
! pip install pandas
```

    fatal: destination path 'PaddleOCR' already exists and is not an empty directory.
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting pip
    [?25l  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a4/6d/6463d49a933f547439d6b5b98b46af8742cc03ae83543e4d7688c2420f8b/pip-21.3.1-py3-none-any.whl (1.7MB)
    [K     |████████████████████████████████| 1.7MB 2.3MB/s eta 0:00:01
    [?25hInstalling collected packages: pip
      Found existing installation: pip 19.2.3
        Uninstalling pip-19.2.3:
          Successfully uninstalled pip-19.2.3
    Successfully installed pip-21.3.1
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting layoutparser==0.0.0
      Downloading https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl (19.1 MB)
         |████████████████████████████████| 19.1 MB 8.7 MB/s            
    [?25hRequirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from layoutparser==0.0.0) (1.20.3)
    Requirement already satisfied: pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from layoutparser==0.0.0) (7.1.2)
    Collecting iopath
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/af/20/65dd9bd25a1eb7fa35b5ae38d289126af065f8a0c1f6a90564f4bff0f89d/iopath-0.1.9-py3-none-any.whl (27 kB)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from layoutparser==0.0.0) (4.27.0)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from layoutparser==0.0.0) (5.1.2)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from layoutparser==0.0.0) (1.1.5)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from layoutparser==0.0.0) (4.1.1.26)
    Collecting portalocker
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/63/eb/f84872af6e9312ea2f345b218015a41191cfd37eeba4a4fd228f241c2a75/portalocker-2.3.2-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->layoutparser==0.0.0) (2.8.2)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->layoutparser==0.0.0) (2019.3)
    Requirement already satisfied: six>=1.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from python-dateutil>=2.7.3->pandas->layoutparser==0.0.0) (1.16.0)
    Installing collected packages: portalocker, iopath, layoutparser
    Successfully installed iopath-0.1.9 layoutparser-0.0.0 portalocker-2.3.2
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting shapely
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ae/20/33ce377bd24d122a4d54e22ae2c445b9b1be8240edb50040b40add950cd9/Shapely-1.8.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.1 MB)
         |████████████████████████████████| 1.1 MB 15.9 MB/s            
    [?25hCollecting scikit-image
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9a/44/8f8c7f9c9de7fde70587a656d7df7d056e6f05192a74491f7bc074a724d0/scikit_image-0.19.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.3 MB)
         |████████████████████████████████| 13.3 MB 63.8 MB/s            
    [?25hCollecting imgaug==0.4.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/66/b1/af3142c4a85cba6da9f4ebb5ff4e21e2616309552caca5e8acefe9840622/imgaug-0.4.0-py2.py3-none-any.whl (948 kB)
         |████████████████████████████████| 948 kB 10.3 MB/s            
    [?25hCollecting pyclipper
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c5/fa/2c294127e4f88967149a68ad5b3e43636e94e3721109572f8f17ab15b772/pyclipper-1.3.0.post2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (603 kB)
         |████████████████████████████████| 603 kB 63.7 MB/s            
    [?25hCollecting lmdb
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4d/cf/3230b1c9b0bec406abb85a9332ba5805bdd03a1d24025c6bbcfb8ed71539/lmdb-1.3.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (298 kB)
         |████████████████████████████████| 298 kB 59.0 MB/s            
    [?25hRequirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleOCR/requirements.txt (line 6)) (4.27.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleOCR/requirements.txt (line 7)) (1.20.3)
    Requirement already satisfied: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleOCR/requirements.txt (line 8)) (2.2.0)
    Collecting python-Levenshtein
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2a/dc/97f2b63ef0fa1fd78dcb7195aca577804f6b2b51e712516cc0e902a9a201/python-Levenshtein-0.12.2.tar.gz (50 kB)
         |████████████████████████████████| 50 kB 2.0 MB/s             
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting opencv-contrib-python==4.4.0.46
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/08/51/1e0a206dd5c70fea91084e6f43979dc13e8eb175760cc7a105083ec3eb68/opencv_contrib_python-4.4.0.46-cp37-cp37m-manylinux2014_x86_64.whl (55.7 MB)
         |████████████████████████████████| 55.7 MB 61.6 MB/s            
    [?25hRequirement already satisfied: cython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleOCR/requirements.txt (line 11)) (0.29)
    Collecting lxml
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7b/01/16a9b80c8ce4339294bb944f08e157dbfcfbb09ba9031bde4ddf7e3e5499/lxml-4.7.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (6.4 MB)
         |████████████████████████████████| 6.4 MB 6.5 MB/s            
    [?25hCollecting premailer
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b1/07/4e8d94f94c7d41ca5ddf8a9695ad87b888104e2fd41a35546c1dc9ca74ac/premailer-3.10.0-py2.py3-none-any.whl (19 kB)
    Requirement already satisfied: openpyxl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleOCR/requirements.txt (line 14)) (3.0.5)
    Requirement already satisfied: scipy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (1.6.3)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (2.2.3)
    Requirement already satisfied: imageio in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (2.6.1)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (4.1.1.26)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (1.16.0)
    Requirement already satisfied: Pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (7.1.2)
    Collecting tifffile>=2019.7.26
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d8/38/85ae5ed77598ca90558c17a2f79ddaba33173b31cf8d8f545d34d9134f0d/tifffile-2021.11.2-py3-none-any.whl (178 kB)
         |████████████████████████████████| 178 kB 93.2 MB/s            
    [?25hCollecting PyWavelets>=1.1.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a1/9c/564511b6e1c4e1d835ed2d146670436036960d09339a8fa2921fe42dad08/PyWavelets-1.2.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (6.1 MB)
         |████████████████████████████████| 6.1 MB 7.7 MB/s            
    [?25hRequirement already satisfied: packaging>=20.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image->-r PaddleOCR/requirements.txt (line 2)) (21.3)
    Requirement already satisfied: networkx>=2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image->-r PaddleOCR/requirements.txt (line 2)) (2.4)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.22.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.1.5)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.21.0)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.0.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.1.1)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (0.8.53)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (4.0.1)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (0.7.1.1)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->-r PaddleOCR/requirements.txt (line 8)) (3.14.0)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from python-Levenshtein->-r PaddleOCR/requirements.txt (line 9)) (56.2.0)
    Collecting cssutils
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/24/c4/9db28fe567612896d360ab28ad02ee8ae107d0e92a22db39affd3fba6212/cssutils-2.3.0-py3-none-any.whl (404 kB)
         |████████████████████████████████| 404 kB 69.6 MB/s            
    [?25hRequirement already satisfied: cachetools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from premailer->-r PaddleOCR/requirements.txt (line 13)) (4.0.0)
    Collecting cssselect
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3b/d4/3b5c17f00cce85b9a1e6f91096e1cc8e8ede2e1be8e96b87ce1ed09e92c5/cssselect-1.1.0-py2.py3-none-any.whl (16 kB)
    Requirement already satisfied: et-xmlfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from openpyxl->-r PaddleOCR/requirements.txt (line 14)) (1.0.1)
    Requirement already satisfied: jdcal in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from openpyxl->-r PaddleOCR/requirements.txt (line 14)) (1.4.1)
    Requirement already satisfied: pycodestyle<2.9.0,>=2.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.8.0)
    Requirement already satisfied: importlib-metadata<4.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->-r PaddleOCR/requirements.txt (line 8)) (4.2.0)
    Requirement already satisfied: pyflakes<2.5.0,>=2.4.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.4.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->-r PaddleOCR/requirements.txt (line 8)) (0.6.1)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->-r PaddleOCR/requirements.txt (line 8)) (0.16.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->-r PaddleOCR/requirements.txt (line 8)) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.1.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.11.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2019.3)
    Requirement already satisfied: decorator>=4.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from networkx>=2.2->scikit-image->-r PaddleOCR/requirements.txt (line 2)) (4.4.2)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from packaging>=20.0->scikit-image->-r PaddleOCR/requirements.txt (line 2)) (3.0.6)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->-r PaddleOCR/requirements.txt (line 8)) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->-r PaddleOCR/requirements.txt (line 8)) (3.9.9)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (1.1.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->imgaug==0.4.0->-r PaddleOCR/requirements.txt (line 3)) (2.8.2)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.4.10)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->-r PaddleOCR/requirements.txt (line 8)) (5.1.2)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->-r PaddleOCR/requirements.txt (line 8)) (0.10.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.0.1)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->-r PaddleOCR/requirements.txt (line 8)) (16.7.9)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.3.4)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.3.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->-r PaddleOCR/requirements.txt (line 8)) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->-r PaddleOCR/requirements.txt (line 8)) (1.25.6)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.8)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata<4.3->flake8>=3.7.9->visualdl->-r PaddleOCR/requirements.txt (line 8)) (3.6.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata<4.3->flake8>=3.7.9->visualdl->-r PaddleOCR/requirements.txt (line 8)) (4.0.1)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl->-r PaddleOCR/requirements.txt (line 8)) (2.0.1)
    Building wheels for collected packages: python-Levenshtein
      Building wheel for python-Levenshtein (setup.py) ... [?25ldone
    [?25h  Created wheel for python-Levenshtein: filename=python_Levenshtein-0.12.2-cp37-cp37m-linux_x86_64.whl size=171680 sha256=29d19da26e04ddae715909d61e9c19df5ac50c4ca6d55ac8307f0de7344d4574
      Stored in directory: /home/aistudio/.cache/pip/wheels/38/b9/a4/3729726160fb103833de468adb5ce019b58543ae41d0b0e446
    Successfully built python-Levenshtein
    Installing collected packages: tifffile, PyWavelets, shapely, scikit-image, lxml, cssutils, cssselect, python-Levenshtein, pyclipper, premailer, opencv-contrib-python, lmdb, imgaug
    Successfully installed PyWavelets-1.2.0 cssselect-1.1.0 cssutils-2.3.0 imgaug-0.4.0 lmdb-1.3.0 lxml-4.7.1 opencv-contrib-python-4.4.0.46 premailer-3.10.0 pyclipper-1.3.0.post2 python-Levenshtein-0.12.2 scikit-image-0.19.1 shapely-1.8.0 tifffile-2021.11.2
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.1.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas) (2019.3)
    Requirement already satisfied: numpy>=1.15.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas) (1.20.3)
    Requirement already satisfied: six>=1.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from python-dateutil>=2.7.3->pandas) (1.16.0)


# 三、通用模型测试
应用现有模型对目标文件进行初步测试，发现不足，制定优化方案。

**发现的问题：**
1. 采用中文OCR模型测试结果发现，由于书写不规范，小数点过大产生了数字被识别为类似汉字或英文字母的情况。
2. 由于原始图片使用普通笔记本进行记录，行的划分效果较好，列的划分效果不好。
3. 由于使用普通笔记本含有日期、星期等信息影响识别准确率。

**解决方法：**

1. 提取关键信息降低不规范字母、汉字等产生的噪声
2. 使用横竖线完整表格，提升表格结构预测精度
3. 将字母、汉字、数字分类简化为数字，减少分类增加文本识别准确度




```python
# 切换到工作目录
import os
os.chdir('/home/aistudio/PaddleOCR/ppstructure')
```


```python
# 下载模型
! mkdir inference && cd inference
# 下载超轻量级表格英文OCR模型的检测模型并解压
! wget -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar && cd inference && tar xf ch_PP-OCRv2_det_infer.tar && cd ..
# 下载超轻量级表格英文OCR模型的识别模型并解压
#! wget -P ./inference/ https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar && cd inference && tar xf ch_PP-OCRv2_rec_infer.tar && cd ..
! wget -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar && cd inference && tar xf en_number_mobile_v2.0_rec_infer.tar && cd ..
# 下载超轻量级英文表格英寸模型并解压
! wget -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && cd inference && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar && cd ..
```

    mkdir: cannot create directory ‘inference’: File exists
    --2022-01-07 09:58:52--  https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
    Resolving paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)... 182.61.200.195, 182.61.200.229, 2409:8c04:1001:1002:0:ff:b001:368a
    Connecting to paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)|182.61.200.195|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3190272 (3.0M) [application/x-tar]
    Saving to: ‘./inference/ch_PP-OCRv2_det_infer.tar.7’
    
    ch_PP-OCRv2_det_inf 100%[===================>]   3.04M  10.9MB/s    in 0.3s    
    
    2022-01-07 09:58:52 (10.9 MB/s) - ‘./inference/ch_PP-OCRv2_det_infer.tar.7’ saved [3190272/3190272]
    
    --2022-01-07 09:58:52--  https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar
    Resolving paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)... 182.61.200.229, 182.61.200.195, 2409:8c04:1001:1002:0:ff:b001:368a
    Connecting to paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)|182.61.200.229|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2699264 (2.6M) [application/x-tar]
    Saving to: ‘./inference/en_number_mobile_v2.0_rec_infer.tar.6’
    
    en_number_mobile_v2 100%[===================>]   2.57M  6.19MB/s    in 0.4s    
    
    2022-01-07 09:58:53 (6.19 MB/s) - ‘./inference/en_number_mobile_v2.0_rec_infer.tar.6’ saved [2699264/2699264]
    
    --2022-01-07 09:58:53--  https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar
    Resolving paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)... 182.61.200.195, 182.61.200.229, 2409:8c04:1001:1002:0:ff:b001:368a
    Connecting to paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)|182.61.200.195|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 19667456 (19M) [application/x-tar]
    Saving to: ‘./inference/en_ppocr_mobile_v2.0_table_structure_infer.tar.7’
    
    en_ppocr_mobile_v2. 100%[===================>]  18.76M  20.7MB/s    in 0.9s    
    
    2022-01-07 09:58:54 (20.7 MB/s) - ‘./inference/en_ppocr_mobile_v2.0_table_structure_infer.tar.7’ saved [19667456/19667456]
    



```python
# 先是输入图像

import cv2
from matplotlib import pyplot as plt
%matplotlib inline

# 读取表格图像并显示
img = cv2.imread('/home/aistudio/6.jpg')
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7f0aee4ed790>




![png](output_6_1.png)



```python
# https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppstructure/table/predict_table.py#L55

from table.predict_table import TableSystem,to_excel
from utility import init_args
# 初始化参数
args = init_args().parse_args(args=[])
args.det_model_dir='inference/ch_PP-OCRv2_det_infer'
#args.rec_model_dir='inference/ch_PP-OCRv2_rec_infer'
args.table_model_dir='inference/en_ppocr_mobile_v2.0_table_structure_infer'
args.rec_model_dir='inference/en_number_mobile_v2.0_rec_infer'
args.image_dir='/home/aistudio/6.jpg'
#args.rec_char_dict_path='../ppocr/utils/ppocr_keys_v1.txt'
args.rec_char_dict_path='../ppocr/utils/en_dict.txt'
args.table_char_dict_path='../ppocr/utils/dict/table_structure_dict.txt'
args.det_limit_side_len=736
args.det_limit_type='min'
args.output='../output/table'
args.use_gpu=False

# 初始化表格识别系统
table_sys = TableSystem(args)
img = cv2.imread('/home/aistudio/6.jpg')
# 执行表格识别
pred_html = table_sys(img)
# 结果存储到excel文件
to_excel(pred_html,'6.xlsx')
print(pred_html)
```

    [2022/01/09 14:27:02] root DEBUG: dt_boxes num : 90, elapse : 21.094706773757935
    [2022/01/09 14:27:04] root DEBUG: rec_res num  : 90, elapse : 1.472456693649292
    <html><body><table><thead><tr><td>888 Sa Su Th We Fr Mo Tu 43</td><td>2=21.</td><td>Date</td></tr></thead><tbody><tr><td></td><td>3</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>2P 13</td><td>43832</td><td>3.6o</td></tr><tr><td>39</td><td>60 AY</td><td>t.y6 CD 3.73</td></tr><tr><td>C  L:49</td><td>AS442 RC </td><td>376</td></tr><tr><td>D 29</td><td>L344D</td><td>378 </td></tr><tr><td>E 523</td><td>B .35 ON</td><td>887</td></tr><tr><td></td><td>L.S5 .32 0</td><td>289 </td></tr><tr><td>H1/ 4ls</td><td>d</td><td>Z</td></tr><tr><td>A</td><td>468 R 77 Z</td><td>4.1</td></tr><tr><td>B y68</td><td>485 S6</td><td>R 3.97 N</td></tr><tr><td>456 56</td><td></td><td>3.78</td></tr><tr><td></td><td>l.66 .b EA</td><td>XOAANE 3.98</td></tr><tr><td>462 N</td><td>460 L78 </td><td>A1D</td></tr><tr><td>.c0 vivo X60ZEISS 2021/09/22 16:33</td><td>B3 31k52</td><td>397 Na</td></tr></table></body></html>



```python
# 读取excel并显示
import pandas as pd
df = pd.read_excel('6.xlsx').fillna('')
print(df)
```

               888 Sa Su Th We Fr Mo Tu 43       2=21.          Date
    0                                                3              
    1                                                               
    2                                2P 13       43832          3.6o
    3                                   39       60 AY  t.y6 CD 3.73
    4                              C  L:49    AS442 RC           376
    5                                 D 29       L344D           378
    6                                E 523    B .35 ON           887
    7                                       L.S5 .32 0           289
    8                              H1/ 4ls           d             Z
    9                                    A  468 R 77 Z           4.1
    10                               B y68      485 S6      R 3.97 N
    11                              456 56                      3.78
    12                                      l.66 .b EA   XOAANE 3.98
    13                               462 N     460 L78           A1D
    14  .c0 vivo X60ZEISS 2021/09/22 16:33    B3 31k52        397 Na


# 四、数据准备
由于试验记录属于低门槛重复性工作，试验员笔迹变化较大，无法形成大规模数据集，所以通过拼接MNIST手写数字数据集，生成多数据数据集进行识别模块的训练，效果图如下：
![](https://ai-studio-static-online.cdn.bcebos.com/6b9e970d9d7246fcb6b3522a339151f1844c0f02fbb14f77bde38d77799f70bc)



```python
%cd ~
!mkdir dataset 
!mkdir dataset/train
!mkdir dataset/test

import cv2
import random
import numpy as np
from tqdm import tqdm
from paddle.vision.datasets import MNIST

# 加载数据集
mnist_train = MNIST(mode='train', backend='cv2')
mnist_test = MNIST(mode='test', backend='cv2')

# 数据集预处理
datas_train = {}
for i in range(len(mnist_train)):
    sample = mnist_train[i]
    x, y = sample[0], sample[1]

    _sum = np.sum(x, axis=0)
    _where = np.where(_sum > 0)
    x = 255 - x[:, _where[0][0]: _where[0][-1]+1]
    if str(y[0]) in datas_train:
        datas_train[str(y[0])].append(x)
    else:
        datas_train[str(y[0])] = [x]

datas_test = {}
for i in range(len(mnist_test)):
    sample = mnist_test[i]
    x, y = sample[0], sample[1]

    _sum = np.sum(x, axis=0)
    _where = np.where(_sum > 0)
    x = 255 - x[:, _where[0][0]: _where[0][-1]+1]
    if str(y[0]) in datas_test:
        datas_test[str(y[0])].append(x)
    else:
        datas_test[str(y[0])] = [x]

# 图片拼接采样
datas_train_list = []
for num in tqdm(range(0, 999)):
    for _ in range(1000):
        imgs = [255 - np.zeros((28, np.random.randint(10)))]
        for word in str(num):
            index = np.random.randint(0, len(datas_train[word]))
            imgs.append(datas_train[word][index])
            imgs.append(255 - np.zeros((28, np.random.randint(10))))
        img = np.concatenate(imgs, 1)
        cv2.imwrite('dataset/train/%03d_%04d.jpg' % (num, _), img)
        datas_train_list.append('train/%03d_%04d.jpg\t%d\n' % (num, _, num))

datas_test_list = []
for num in tqdm(range(0, 999)):
    for _ in range(50):
        imgs = [255 - np.zeros((28, np.random.randint(10)))]
        for word in str(num):
            index = np.random.randint(0, len(datas_test[word]))
            imgs.append(datas_test[word][index])
            imgs.append(255 - np.zeros((28, np.random.randint(10))))
        img = np.concatenate(imgs, 1)
        cv2.imwrite('dataset/test/%03d_%04d.jpg' % (num, _), img)
        datas_test_list.append('test/%03d_%04d.jpg\t%d\n' % (num, _, num))

# 数据列表生成
with open('dataset/train.txt', 'w') as f:
    for line in datas_train_list:
        f.write(line)

with open('dataset/test.txt', 'w') as f:
    for line in datas_test_list:
        f.write(line)
```

    /home/aistudio
    mkdir: cannot create directory ‘dataset’: File exists
    mkdir: cannot create directory ‘dataset/train’: File exists
    mkdir: cannot create directory ‘dataset/test’: File exists


    100%|██████████| 999/999 [02:20<00:00,  7.39it/s]
    100%|██████████| 999/999 [00:06<00:00, 152.41it/s]


# 五、原理详解

### 整体pipeline介绍

PP-Structure 的表格识别模型算法属于基于端到端的方法

表格识别算法由三个模型组成：
1. 文字检测模型：用于检测表格里的文本
2. 文字识别模型：用于对检测到的文本进行识别
3. 表格单元格预测和表格结构预测模型：用于预测表格结构的HTML信息和表格单元格坐标

三个模型的串联过程如下图所示：

<center class="img">
<img src="https://ai-studio-static-online.cdn.bcebos.com/07fad4f0bc6a473f9258d913a9afc380c3cd582cc44f4d0fa4cdbade934e07b5" width="1300"/></center>
<center>图 1：表格识别pipeline</center>


具体过程为：
1. 使用文字检测模型用于检测表格里的文本
2. 使用文字识别模型对检测到的文本进行识别，到这一步，我们拿到了文字的框和文字信息
3. 使用表格单元格预测和表格结构预测模型进行单元格坐标预测和表格结构的HTML信息预测
4. 对2中的文字框和3中的单元格坐标进行聚合，如下图所示，根据<font color="#dd0000">红色的文字检测框和蓝色的单元格坐标检测框之间的IOU</font>进行判定是否需要聚合。
5. 在完成文本框聚合之后，对文本框进行一个从上到下，从左到右的排序，根据排序后文本框的索引即可拿到对应的文字信息，然后文字信息做一个<font color="#dd0000">字符串拼接</font>即可得到最终单元格里的文本内容。

<center class="img">
<img src="https://ai-studio-static-online.cdn.bcebos.com/32a7368a59f142dcb735247fa7537ae1681c5541f92444388bd916a942fcdfa5" width="1300"/></center>
<center>图 2：文字框和单元格坐标聚合示意图</center>

# 六、模型训练

训练表格识别，需要训练三个模型，分别为文本检测，文本识别，表格结构模型


```python
%cd ~/PaddleOCR

!wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar
!cd pretrain_models && tar -xf ch_ppocr_mobile_v2.0_rec_pre.tar && rm -rf ch_ppocr_mobile_v2.0_rec_pre.tar
```

    /home/aistudio/PaddleOCR
    --2022-01-07 10:41:34--  https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar
    Resolving paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)... 182.61.200.229, 182.61.200.195, 2409:8c04:1001:1002:0:ff:b001:368a
    Connecting to paddleocr.bj.bcebos.com (paddleocr.bj.bcebos.com)|182.61.200.229|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 16130750 (15M) [application/x-tar]
    Saving to: ‘./pretrain_models/ch_ppocr_mobile_v2.0_rec_pre.tar’
    
    ch_ppocr_mobile_v2. 100%[===================>]  15.38M  8.75MB/s    in 1.8s    
    
    2022-01-07 10:41:37 (8.75 MB/s) - ‘./pretrain_models/ch_ppocr_mobile_v2.0_rec_pre.tar’ saved [16130750/16130750]
    



```python
%cd ~/PaddleOCR

!python tools/train.py -c ../multi_mnist.yml
```

    [2022/01/07 22:16:48] root INFO: best metric, acc: 0.9883683485812144, norm_edit_dis: 0.9959676343816957, fps: 7665.377555879139, best_epoch: 8


```python
%cd ~/PaddleOCR

!python3 tools/export_model.py \
    -c ../multi_mnist.yml -o Global.pretrained_model=../output/multi_mnist/best_accuracy \
    Global.load_static_weights=False \
    Global.save_inference_dir=../inference/multi_mnist
```

    /home/aistudio/PaddleOCR
    W0108 08:54:24.882784   482 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0108 08:54:24.887475   482 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [2022/01/08 08:54:29] root INFO: load pretrain successful from ../output/multi_mnist/best_accuracy
    [2022/01/08 08:54:31] root INFO: inference model is saved to ../inference/multi_mnist/inference


# 七、模型推理
对整体识别错误率高的图片进行分块识别，以提高准确率，最后将分块识别的数据汇总到一张excel表中，以备后续的试验统计。下面以其中的一组试验数据为例进行识别。


```python
#图片处理
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
path = "/home/aistudio/6.jpg" # 图片路径
image = Image.open(path) #读取图片
image = np.array(image)
#裁剪图片，保留关键数字信息
x = 730
y = 450
imageR = image[x:2000,y:2600,]
plt.imshow(image)
plt.show()
plt.imshow(imageR)
plt.show()
imageR = Image.fromarray(imageR, 'RGB') 
#对裁剪数据进行网格分界，提高检测准确率
draw = ImageDraw.Draw(imageR)
draw.line((65,0) + (65,3500),fill = 128,width = 22)
draw.line((550,0) + (550,3500),fill = 128,width = 22)#线的起点和终点，线宽
draw.line((1100,0) + (1100,3500),fill = 128,width = 22)
draw.line((1600,0) + (1600,3500),fill = 128,width = 22)
for i in [200,400,600,800,1000,1202]:
        draw.line((0,i) + (2600,i),fill = 128,width = 22)
plt.imshow(imageR)
plt.show()
imageR.save('/home/aistudio/66.jpg')
```


![png](output_18_0.png)



![png](output_18_1.png)



![png](output_18_2.png)



```python
# https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppstructure/table/predict_table.py#L55

from table.predict_table import TableSystem,to_excel
from utility import init_args
# 初始化参数
args = init_args().parse_args(args=[])
args.det_model_dir='/home/aistudio/inference/ch_PP-OCRv2_det_infer'
#args.rec_model_dir='inference/ch_PP-OCRv2_rec_infer'
args.table_model_dir='/home/aistudio/inference/en_ppocr_mobile_v2.0_table_structure_infer'
##args.rec_model_dir='inference/en_number_mobile_v2.0_rec_infer'
args.rec_model_dir='/home/aistudio/inference/multi_mnist'

args.image_dir='/home/aistudio/66.jpg'
#args.rec_char_dict_path='../ppocr/utils/ppocr_keys_v1.txt'
##args.rec_char_dict_path='../ppocr/utils/en_dict.txt'
args.rec_char_dict_path='/home/aistudio/label_list.txt'

args.table_char_dict_path='/home/aistudio/PaddleOCR/ppocr/utils/dict/table_structure_dict.txt'
args.det_limit_side_len=372
args.det_limit_type='max'
args.output='../output/table'
args.use_gpu=False

# 初始化表格识别系统
table_sys = TableSystem(args)
img = cv2.imread('/home/aistudio/66.jpg')
# 执行表格识别
pred_html = table_sys(img)
# 结果存储到excel文件
to_excel(pred_html,'6.xlsx')
print(pred_html)
```

    [2022/01/09 19:09:51] root DEBUG: dt_boxes num : 24, elapse : 0.1392822265625
    [2022/01/09 19:09:52] root DEBUG: rec_res num  : 24, elapse : 0.4081110954284668
    <html><body><table><thead><tr><td>429</td><td>438</td><td>432</td><td>360</td></tr></thead><tbody><tr><td>439</td><td>460</td><td>446</td><td>323</td></tr><tr><td>649</td><td>454</td><td>442</td><td>386</td></tr><tr><td>337</td><td>443</td><td>440</td><td>378</td></tr><tr><td>755</td><td>461</td><td>435</td><td>382</td></tr><tr><td>447</td><td>445</td><td>432</td><td>788</td></tr></table></body></html>



```python
# 读取excel并显示
import pandas as pd
df = pd.read_excel('6.xlsx').fillna('')
print(df)
```

       429  438  432  360
    0  439  460  446  323
    1  649  454  442  386
    2  337  443  440  378
    3  755  461  435  382
    4  447  445  432  788


# 八、结论
通用模型测试效果可以查看项目第三部分（由于识别率过低，这里不附图片），通过模型优化后效果如下图对比：


![](https://ai-studio-static-online.cdn.bcebos.com/78788a2f999f4b6b9956ccb7fabcfb4cefa5eda02483461bb6a880a3dce700b1)
![](https://ai-studio-static-online.cdn.bcebos.com/c7b4a9c0d52747028d0b813af67d2d134e491b1f0c13408fae09ae1352f1b87b)

通过对比可以看出虽然经过优化后识别效果大幅提升，识别率如下表

| 对比项| 准确率 | |
| -------- | -------- | -------- |
|行数据   | 17%     | 
| 列数据    | 50%    | 
| 组数据    | 71 %   | 
| 单个数据  | 86%   | 

通过以上数据分析对比可知：

1、行数据准确率较低由于笔记本自带横线与手写数据较差造成识别错误。

2、列数据识别效果可以看出，中间两列识别率100%，两侧数据由于边缘导致识别率降低。

3、单个数据识别率较高说明通过MNIST训练的模型通用行较强。

工业落地解决办法：

1、更换纸质记录表格，使用标准统一的表格记录数据。

2、增加单个表格书写空间，降低数字溢的影响。



# 九、致谢

1、感谢寂寞你快进去的项目[PaddleOCR：基于 MNIST 数据集的手写多数字识别](https://aistudio.baidu.com/aistudio/projectdetail/1847732?channelType=0&channel=0)

2、感谢飞桨的[动手学OCR十讲](https://aistudio.baidu.com/aistudio/education/group/info/25207)

3、感谢[飞桨领航团AI达人养成营](https://aistudio.baidu.com/aistudio/education/group/info/25038)

# 十、个人简介
一名从事创新创业教育工作的普通教师。

希望分享能给您的学习提供帮助，希望Rechard Stallman提出的开源精神可以让AI技术发展的越来越好，也希望自主可控的飞桨框架能够助力中国数字经济飞速发展。
