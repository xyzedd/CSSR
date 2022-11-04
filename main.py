
import numpy as np
import argparse

import dataset
import json
import metrics
import methods.cssr
import methods.cssr_ft

from methods import *
import os
import sys
import methods.util as util

import warnings
 
warnings.filterwarnings('ignore')


def save_everything(subfix = ""):
    # save model
    if subfix == "":
        mth.save_model(saving_path + 'model.pth')
    # save training process data
    with open(saving_path + "hist.json",'w') as f:
        json.dump(history, f)

def load_everything(subfix = ""):
    global history,best_auroc,best_acc
    # load the model
    if subfix != "" :
        mth.load_model(saving_path + f'model_{subfix}.pth')
    else:
        mth.load_model(saving_path + 'model.pth')
    # load history
    history = np.load(saving_path + 'hist.npy',allow_pickle=True).tolist()
    
    bac,bau = get_best_acc_auc()
    best_acc = bac[1]
    best_auroc = bau[1]

def log_history(epoch,data_dict):
    item = {
        'epoch' : epoch
    }
    item.update(data_dict)
    if isinstance(history,list):
        history.append(item)
    print(f"Epoch:{epoch}")
    for key in data_dict.keys():
        print("  ",key,":",data_dict[key])

best_acc = -1
best_auroc = -1
last_acc = -1
last_auroc = -1
last_f1 = -1
cwauc = -1

def training_main():
    tot_epoch = config['epoch_num']
    global best_acc,best_auroc

    for epoch in range(mth.epoch,tot_epoch):
        sys.stdout.flush()
        losses = mth.train_epoch()
        acc = 0
        auroc = 0
        if epoch % 1 == 0:
            save_everything(f'ckpt{epoch}')

        if epoch % test_interval == test_interval - 1 :
            # big test with aurocs
            scores,thresh,pred = mth.knownpred_unknwonscore_test(test_loader)
            acc = evaluation.close_accuracy(pred)
            open_detection = evaluation.open_detection_indexes(scores,thresh)
            auroc = open_detection['auroc']
            log_history(epoch,{
                "loss" : losses,
                "close acc" : acc,
                "open_detection" : open_detection,
                "open_reco" : evaluation.open_reco_indexes(scores,thresh,pred)
            })
        else:
            # close_pred = mth.known_prediction_test(train_labeled_loader,train_unlabeled_loader,test_loader)
            # acc = evaluation.close_accuracy(close_pred)
            log_history(epoch,{
                "loss" : losses,
                # "close acc" : acc,
            })
        # if epoch % 10 == 0:
        save_everything()
        if acc > best_acc:
            best_acc = acc
            save_everything("acc")
        if auroc > best_auroc:
            best_auroc = auroc
            save_everything("auroc")

def get_best_acc_auc():
    best_auc,best_acc = [0,0],[0,0]
    for itm in history:
        epoch = itm['epoch']
        if 'close acc' in itm.keys():
            acc = itm['close acc']
            if acc > best_acc[1]:
                best_acc = [epoch,acc]
        if not 'open_detection' in itm.keys():
            continue
        auc = itm['open_detection']['auroc']
        if auc > best_auc[1]:
            best_auc = [epoch,auc]
    return best_acc,best_auc

def overall_testing():
    global train_loader,test_loader
    global last_acc,last_auroc,last_f1,cwauc,best_acc,best_auroc

    scores,thresh,pred = mth.knownpred_unknwonscore_test(test_loader)
    last_acc = evaluation.close_accuracy(pred)
    indexes = evaluation.open_detection_indexes(scores,thresh)
    last_auroc = indexes['auroc']
    osr_indexes = evaluation.open_reco_indexes(scores,thresh,pred)
    last_f1 = osr_indexes['macro_f1']
    log_history(-1,{
        "close acc" : last_acc,
        "open_detection" :indexes,
        "open_reco" : osr_indexes
    })
    print("Metrics", {\
        "close acc" : last_acc,
        "open_detection" :indexes,
        "open_reco" : osr_indexes})
        


def update_config_keyvalues(config,update):
    if update == "":
        return config
    spls = update.split(",")
    for spl in spls:
        key,val = spl.split(':')
        key_parts = key.split('.')
        sconfig = config
        for i in range(len(key_parts) - 1):
            sconfig = sconfig[key_parts[i]]
        org = sconfig[key_parts[-1]]
        if isinstance(org,bool):
            sconfig[key_parts[-1]] = val == 'True'
        elif isinstance(org,int):
            sconfig[key_parts[-1]] = int(val)
        elif isinstance(org,float):
            sconfig[key_parts[-1]] = float(val)
        else:
            sconfig[key_parts[-1]] = val
        print("Updating",key,"with",val,"results in",sconfig[key_parts[-1]])
    return config

def update_subconfig(cfg,u):
    for k in u.keys():
        if not k in cfg.keys() or not isinstance(cfg[k],dict):
            cfg[k] = u[k]
        else:
            update_subconfig(cfg[k],u[k])
        
def load_config(file):
    with open(file,"r") as f :
        config = json.load(f)
    if 'inherit' in config.keys():
        inheritfile = config['inherit']
        if inheritfile != 'None':
            parent = load_config(inheritfile)
            update_subconfig(parent,config)
            config = parent
    return config

def set_up_gpu(args):
    if args.gpu != 'cpu':
        args.gpu =  ",".join([c for c in args.gpu])
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

if __name__ == "__main__":
    import torch
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=False,default="1", help='GPU number')
    parser.add_argument('--ds', type=str, required=False,default="None", help='dataset setting, choose file from ./exps')
    parser.add_argument('--config', type=str, required=False,default="None", help='model configuration, choose from ./configs')
    parser.add_argument('--save', type=str, required=False,default="None", help='Saving folder name')
    parser.add_argument('--method', type=str, required=False,default="ours", help='Methods : ' + ",".join(util.method_list.keys()))
    parser.add_argument('--test', action="store_true",help='Evaluation mode')
    parser.add_argument('--configupdate', type=str, required=False,default="", help='Update several key values in config')
    parser.add_argument('--test_interval', type=int, required=False,default=1, help='The frequency of model evaluation')
    
    args = parser.parse_args()

    test_interval = args.test_interval
    if not args.save.endswith("/"):
        args.save += "/"
    
    set_up_gpu(args)
    
    saving_path = "./save/" + args.save
    util.setup_dir(saving_path)

    if args.config != "None" :
        config = load_config(args.config)
    else:
        config = {}
    config = update_config_keyvalues(config,args.configupdate)
    args.bs = config['batch_size']
    print('Config:',config)
    
    train_loader , test_loader ,classnum = dataset.load_partitioned_dataset(args,args.ds)
    mth = util.method_list[args.method](config,classnum,train_loader.dataset)
    
    history = []
    evaluation = metrics.OSREvaluation(test_loader)
    
    if not args.test:
        print(f"TotalEpochs:{config['epoch_num']}")
        training_main()
        save_everything()
        overall_testing()
        print("Overall: LastAcc",last_acc," LastAuroc", last_auroc," BestAcc",best_acc," BestAuroc",best_auroc,"CWAuroc",cwauc)
    else:
        load_everything()
        overall_testing()

