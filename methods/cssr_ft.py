
from methods.augtools import  HighlyCustomizableAugment, RandAugmentMC
import methods.util as util
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import random
from methods.util import AverageMeter
import time
from torchvision.transforms import transforms
from methods.cssr import Backbone,AutoEncoder

class LinearClassifier(nn.Module):

    def __init__(self,inchannels,num_class, config):
        super().__init__()
        if config['projection_dim'] > 0:
            self.proj = nn.Sequential(
                nn.Conv2d(inchannels, config['projection_dim'], 1,padding= 0, bias=False),
                nn.BatchNorm2d(config['projection_dim']),
                nn.LeakyReLU(inplace=True))
            inchannels = config['projection_dim'] 
        else:
            self.proj = nn.Identity()
        self.cls = nn.Conv2d(inchannels, num_class , 1,padding= 0, bias=False)
    
    
    def forward(self,x):
        x = self.proj(x)
        x1 = self.cls(x)
        return x,x1


class CSSRClassifier(nn.Module):

    def __init__(self,inchannels,num_class, config):
        super().__init__()
        if config['projection_dim'] > 0:
            self.proj = nn.Sequential(
                nn.Conv2d(inchannels, config['projection_dim'], 1,padding= 0, bias=False),
                nn.BatchNorm2d(config['projection_dim']),
                nn.LeakyReLU(inplace=True))
            inchannels = config['projection_dim'] 
        else:
            self.proj = nn.Identity()
                
        ae_hidden = config['ae_hidden']
        ae_latent = config['ae_latent']
        self.class_aes = []
        for i in range(num_class):
            ae = AutoEncoder(inchannels,ae_hidden,ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        self.useL1 = config['error_measure'] == 'L1'

        self.reduction = -1 if config['model'] == 'pcssr' else 1
        self.reduction *= config['gamma']

    
    def ae_error(self,rc,x):
        if self.useL1:
            return torch.norm(rc - x,p = 1,dim = 1,keepdim=True) * self.reduction
        else:
            return torch.norm(rc - x,p = 2,dim = 1,keepdim=True) ** 2 * self.reduction

    clip_len = 100

    def forward(self,x):
        x = self.proj(x)
        cls_ers = []
        for ae in self.class_aes:
            rc,_ = ae(x)
            cls_er = self.ae_error(rc,x)
            if CSSRClassifier.clip_len > 0:
                cls_er = torch.clamp(cls_er,-CSSRClassifier.clip_len,CSSRClassifier.clip_len)
            cls_ers.append(cls_er)
        logits = torch.cat(cls_ers,dim=1) 
        return x,logits



class BaselineModel(nn.Module):

    def __init__(self,num_classes,config,crt):
        super().__init__()
        self.backbone = Backbone(config,3)
        self.crt = crt

        clsblock = {'linear':LinearClassifier, 'pcssr':CSSRClassifier,'rcssr' : CSSRClassifier}
        mod_config = config['category_model']
        self.cls = clsblock[mod_config['model']](self.backbone.output_dim,num_classes,mod_config).cuda()
        self.config = config
        
    def forward(self,x,ycls = None,fixbackbone=False):
        if fixbackbone:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)

        def pred_score(xcls):
            score_reduce = lambda x : x.reshape([x.shape[0],-1]).mean(axis = 1)
            
            probs = self.crt(xcls,prob = True).cpu().numpy()
            pred = probs.argmax(axis = 1)
            max_prob = probs.max(axis = 1)

            cls_scores = xcls.cpu().numpy()[[i for i in range(pred.shape[0])],pred]
            rep_scores = torch.abs(x.detach()).mean(dim = 1).cpu().numpy()
            R = [cls_scores,rep_scores,0,0,max_prob]

            scores = score_reduce(eval(self.config['score']))
            return pred,scores

        if self.training:
            x,logitcls = self.cls(x)
            return logitcls
        else:
            x,xcls = self.cls(x)
            pred,scores = pred_score(xcls)
            return pred,scores
        

class CSSRCriterion(nn.Module):

    def get_onehot_label(self,y,clsnum):
        y = torch.reshape(y,[-1,1])
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)

    def __init__(self,avg_order,enable_sigma = True):
        super().__init__()
        self.avg_order = {"avg_softmax":1,"softmax_avg":2}[avg_order]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enable_sigma = enable_sigma

    def forward(self,x,y = None, prob = False,pred = False,dontreduce = False):
        if self.avg_order == 1:
            g = self.avg_pool(x).view(x.shape[0],-1)
            g = torch.softmax(g,dim=1)
        elif self.avg_order == 2:
            g = torch.softmax(x,dim=1)
            g = self.avg_pool(g).view(x.size(0), -1)
        if prob: return g
        if pred: return torch.argmax(g,dim = 1)
        if not dontreduce:
            loss = -torch.sum(self.get_onehot_label(y,g.shape[1]) * torch.log(g),dim=1).mean()
        else:
            loss = -torch.sum(self.get_onehot_label(y,g.shape[1]) * torch.log(g),dim=1)

        return loss


def manual_contrast(x):
    s = random.uniform(0.1,2)
    return x * s


class WrapDataset(data.Dataset):

    def __init__(self,labeled_ds,config,inchan_num = 3) -> None:
        super().__init__()
        self.labeled_ds = labeled_ds

        __mean = [0.5,0.5,0.5][:inchan_num]
        __std = [0.25,0.25,0.25][:inchan_num]

        trans = [transforms.RandomHorizontalFlip()] 
        if config['cust_aug_crop_withresize']:
            trans.append(transforms.RandomResizedCrop(size = util.img_size,scale = (0.25,1)))
        elif util.img_size > 200:
            trans += [transforms.Resize(256),transforms.RandomResizedCrop(util.img_size)]
        else:
            trans.append(transforms.RandomCrop(size=util.img_size,
                                    padding=int(util.img_size*0.125),
                                    padding_mode='reflect'))
        if config['strong_option'] == 'RA':
            trans.append(RandAugmentMC(n=2, m=10))
        elif config['strong_option'] == 'CUST':
            trans.append(HighlyCustomizableAugment(2,10,-1,labeled_ds,config))
        elif config['strong_option'] == 'NONE':
            pass
        else:
            raise NotImplementedError()
        trans += [transforms.ToTensor(),
                  transforms.Normalize(mean=__mean, std=__std)]
        
        if config['manual_contrast']:
            trans.append(manual_contrast)
        strong = transforms.Compose(trans)

        self.simple = transforms.Compose(([transforms.RandomHorizontalFlip()]) + [
                                        transforms.Resize(256),
                                        transforms.RandomResizedCrop(size = util.img_size,scale = (0.25,1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=__mean, std=__std)])
        # self.testaug = gen_testaug_transform()
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=__mean, std=__std)])

        td = {'strong' : strong, 'simple' : self.simple}
        self.aug = td[config['cat_augmentation']]
        self.test_mode = False

    def __len__(self) -> int:
        return len(self.labeled_ds)
    
    def __getitem__(self, index: int) :
        img,lb,_ = self.labeled_ds[index]
        if self.test_mode:
            img = self.normalize(img)
        else:
            img = self.aug(img)
        return img,lb,index

@util.regmethod('cssr_ft')
class CSSRFTMethod:

    def get_cfg(self,key,default):
        return self.config[key] if key in self.config else default
    
    def __init__(self, config, clssnum, train_set) -> None:
        self.config = config
        self.epoch = 0
        self.clsnum = clssnum 
        self.crt = CSSRCriterion(config['arch_type'],False)
        self.model = BaselineModel(self.clsnum,config,self.crt).cuda()
        # ---- Training  Related
        self.batch_size = config['batch_size']
        self.lr = config['learn_rate'] * (self.batch_size / 128)

        self.modelopt = torch.optim.SGD([
            { 'params': self.model.cls.parameters(), 'lr' : self.lr, 'weight_decay':1e-4 },
            ], lr=self.lr,weight_decay=5e-4)
            
        # ---- schedules
        self.lrdecay = self.config['lr_decay']
        self.wrap_ds = WrapDataset(train_set,self.config,3)
        self.wrap_loader = data.DataLoader(self.wrap_ds,
            batch_size=self.config['batch_size'], shuffle=True,pin_memory=True, num_workers=6)
        self.lr_schedule = util.get_scheduler(self.config,self.wrap_loader)

    def train_epoch(self):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        train_acc = AverageMeter()
        running_loss = AverageMeter()
        
        self.model.train()
        self.model.backbone.eval()
        progress_bar = tqdm.tqdm(self.wrap_loader)
        endtime = time.time()
        for i, data in enumerate(progress_bar):
            data_time.update(time.time() - endtime)
            progress_bar.set_description('epoch ' + str(self.epoch))
            self.lr = self.lr_schedule.get_lr(self.epoch,i,self.lr)
            util.set_lr([self.modelopt],self.lr)
            sx, lb = data[0].cuda(),data[1].cuda()
            
            cls_logits = self.model(sx,ycls = lb,fixbackbone=True) 
            loss = self.crt(cls_logits,lb)
            pred = self.crt(cls_logits,pred = True).cpu().numpy()
            
            self.modelopt.zero_grad()
            loss.backward()
            self.modelopt.step()
            
            nplb = data[1].numpy()
            train_acc.update((pred == nplb).sum() / pred.shape[0],pred.shape[0])
            running_loss.update(loss.item())
            batch_time.update(time.time() - endtime)
            endtime = time.time()
            
            progress_bar.set_postfix(
                acc='%.4f' % train_acc.avg,
                loss='%.4f' % running_loss.avg,
                datatime = '%.4f' % data_time.avg,
                batchtime = '%.4f' % batch_time.avg,
                learnrate = '%.4f' % self.lr,
                )
            if i % 200 == 0:
                print("Itr",i,'TrainAcc:%.4f' % train_acc.avg,'loss:%.4f' % (running_loss.avg),'learnrate:%.4f' % self.lr) 
        
        training_res = \
                {"Loss" : running_loss.avg,
                "TrainAcc" : train_acc.avg,
                "Learn Rate" : self.lr,
                "DataTime" : data_time.avg,
                "BatchTime" : batch_time.avg}

        return training_res

    def known_prediction_test(self,test_loader):
        self.model.eval()
        pred,scores = self.scoring(test_loader)
        return pred

    def scoring(self,loader):
        scores = []
        prediction = []
        with torch.no_grad():
            for d in tqdm.tqdm(loader):
                x1 = d[0].cuda(non_blocking = True)
                pred,scr = self.model(x1)
                prediction.append(pred)
                scores.append(scr)

        prediction = np.concatenate(prediction)
        scores = np.concatenate(scores)
        return prediction,scores


    def knownpred_unknwonscore_test(self, test_loader):
        self.model.eval()
        pred,scores = self.scoring(test_loader)
        return scores,-9999999,pred

    def save_model(self,path):
        save_dict = {
            'model' : self.model.state_dict(),
            'config': self.config,
            'optimzer' : self.modelopt.state_dict(),
            'epoch' : self.epoch
        }
        torch.save(save_dict,path)

    def load_model(self,path):
        save_dict = torch.load(path)
        print("The loading model has config")
        print(save_dict['config'])
        self.model.load_state_dict(save_dict['model'])
        if 'optimzer' in save_dict:
            self.modelopt.load_state_dict(save_dict['optimzer'])
        self.epoch = save_dict['epoch']
