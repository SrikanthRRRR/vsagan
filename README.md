# VSAGAN

This repository contains PyTorch implementation of the vsagan based on following paper: Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection

## 1. Table of Contents

## 2. Python 3.7.9 Installation
# Downloading python source:
cd /usr/src
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz

# Extract Python-3.7.9
tar xzf Python-3.7.9.tgz

# Compilation
cd Python-3.7.9
./configure --enable-optimizations
make altinstall

#Cleanup
rm /usr/src/Python-3.7.9.tgz
ln -s /usr/local/bin/python3.7 /usr/bin/python3.7

# Installation
cd /usr/tmp
git clone https://github.com/sraghunathan1612/vsagan.git
/usr/local/bin/python3.7 -m pip install --upgrade pip
/usr/local/bin/python3.7 -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
/usr/local/bin/python3.7 -m pip install --user --requirement requirements.txt
/usr/local/bin/python3.7 -m pip install numba 
/usr/local/bin/python3.7 -m pip install pynvml 

## 3. Experiment
To replicate the results in the paper for CIFAR10  dataset, run the following commands:

## 4. Training
To list the arguments, run the following command:
```
python train.py --dataset=OCT (custom folder name under /data)
```

### 4.1. Training on CIFAR10
To train the model on CIFAR10 dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset cifar10                                                             \
    --niter <number-of-epochs>                                                    \
    --abnormal_class                                                              \
        <airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck>    \
    --display                                   # optional if you want to visualize        
```

### 4.2. Train on Custom Dataset
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

Then model training is the same as the training explained above.

```
python train.py                     \
    --dataset <name-of-the-data>    \
    --isize <image-size>            \
    --niter <number-of-epochs>      \
    --display                       # optional if you want to visualize
```

For more training options, run `python train.py -h`.

## 7. Variations
All methods implemented are parameter driven.
To enable latent space normalization --g_lat_dim > 0 
To add attention layers at encoder, --addCBAM=1, to add attention layers at decoder --addCBAM=2, to disable --addCBAM=0
To add DSC, --DSC=1, else --DSC=0. 
To change resolution of images used, --isize=128/256.
To view the generator and discriminator network, --vwMdl=1
other control parameters and default values can be looked up from options.py

## 6. Reference
[1] S. Akçay, A. Atapour-Abarghouei, T. P. Breckon, Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection, in: International Joint Conference on Neural Networks (IJCNN), IEEE, 2019.
[2] Guoliang Liuy, Shiyong Lan y, Ting Zhangy, Weikang Huangy and Wenwu Wang, SAGAN: SKIP-ATTENTION GAN FOR ANOMALY DETECTION, Proceedings - International Conference on Image Processing, ICIP (2021) 2021-September 2468-2472
[3] https://github.com/elbuco1/CBAM
