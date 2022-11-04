# Class Specific Semantic Reconstruction for Open Set Recognition [TPAMI 2022] 

Official PyTorch implementation of [Class Specific Semantic Reconstruction for Open Set Recognition](https://ieeexplore.ieee.org/document/9864101).

## 1. Train

Before training, please setup dataset directories in `dataset.py`:
```
DATA_PATH = ''          # path for cifar10, svhn
TINYIMAGENET_PATH = ''  # path for tinyimagenet
LARGE_OOD_PATH = ''     # path for ood datasets, e.g., iNaturalist in imagenet experiment
IMAGENET_PATH = ''      # path for imagenet-1k datasets
```

To train models from scratch, run command:
```
python main.py --gpu 0 --ds {DATASET} --config {MODEL} --save {SAVING_NAME} --method cssr
```

Command options: 
- **DATASET:** Experiment configuration file, specifying datasets and random splits, e.g., `./exps/$dataset/spl_$s.json`.
- **MODEL:** OSR model configuration file, specifying model parameters, e.g., ./configs/$model/$dataset.json. `$model` includes linear/pcssr/rcssr, which corresponds to the baseline and the proposed model.

Or simply run bash file `sh run.sh` to run all experiments simultaneously.

To train models by finetuning pretrained backbones, like experiments for imagenet-1k, run command:
```
python main.py --gpu 0 --ds ./exps/imagenet/vs_inaturalist.json --config ./configs/rcssr/imagenet.json --save imagenet1k_rcssr --method cssr_ft
```

## 2. Evaluation

Add `--test` on training commands to restore and evaluate a pretrained model on specified data setup, e.g.,
```
python main.py --gpu 0 --ds {DATASET} --config {MODEL} --save {SAVING_NAME} --method cssr --test
```

With models trained by `sh run.sh`, script `collect_metrics.py` helps collect and present experimental results: `python collect_metrics.py`


## 3. Citation
```
@ARTICLE{9864101,
  author={Huang, Hongzhi and Wang, Yu and Hu, Qinghua and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Class-Specific Semantic Reconstruction for Open Set Recognition},
  year={2022},
  doi={10.1109/TPAMI.2022.3200384}
}
```