# AIM-CIC
This repository contains code and trained models for our paper AIM-CIC.

## Usage
Our code is implemented with PyTorch and Detectron2. We recommand running our code in Linux and using Anaconda to setup Python environment.
1. Download and install [Anaconda](https://www.anaconda.com/products/individual#Downloads)
2. Install [PyTorch](https://pytorch.org/)
3. Install [Detectron2](https://github.com/facebookresearch/detectron2)
4. Clone this repository and run ```python start_app_multiclass.py``` to start local service. After that you can visit [http://localhost:6789/0.5](http://localhost:6789/0.5) to test our demo.

Please notice that this repository only contains the code for our paper's demo, since models' weight files are too large for github to host. If you want to get the weight files, please contact the corresponding author.
Also, we provide an online demo [here](http://8.142.44.158:6789/0.5).