
# Open set recognition implementation
In this project, I have implemented open set recognition using different references. T
There are three branches main, integrate_cac, and mixup. The implementation is performed for only CIFAR10 dataset as of now.

In main branch it consists of the implementation of the autoencoder based CSSR, where distance is measured between the latent representation and its mean. The integrate_cac branch uses a cac loss based technique and the mixup branch uses the manifold mixup technique.

Note: This repository is still in the development phase and you might encounter issues while running the code. Please open an issue for any errors.

 1. Class Specific Semantic Reconstruction for Open Set Recognition [[Paper]](https://ieeexplore.ieee.org/document/9864101), [[Code]](https://github.com/xyzedd/CSSR).
 2. Class Anchor Clustering: a Loss for Distance-based Open Set Recognition [[Paper]](https://arxiv.org/abs/2004.02434), [[Code]](https://github.com/dimitymiller/cac-openset).
 3. Manifold Mixup: Better Representations by Interpolating Hidden States [[Paper]](https://arxiv.org/abs/1806.05236), [[Code]](https://github.com/vikasverma1077/manifold_mixup)


 TODO: 
  1. refactor the code
  2. make arguments consice to run from terminal
  3. Run training for all datasets


Run ```main.py``` for training. And for evaluation set ```args.test``` to ```True```.
