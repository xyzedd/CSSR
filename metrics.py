
import numpy as np
from sklearn import metrics


class OSREvaluation():

    def __init__(self,test_loader) -> None:
        super().__init__()
        labels = test_loader.dataset.labels
        self.test_labels = np.array(labels,np.int)
        
        self.close_samples = self.test_labels >= 0
        self.close_samples_ct = np.sum(self.close_samples)
    
    def close_accuracy(self,prediction):
        return np.sum((prediction == self.test_labels) & self.close_samples) / self.close_samples_ct
    
    
    def open_detection_indexes(self,scores,thresh):
        if np.isnan(scores).any() or np.isinf(scores).any():
            return {"auroc" : -1}
        fpr, tpr, thresholds  =  metrics.roc_curve(self.close_samples, scores) 
        auroc = metrics.auc(fpr,tpr)
        precision, recall, _ = metrics.precision_recall_curve(self.close_samples, scores)
        aupr_in = metrics.auc(recall,precision)
        precision, recall, _ = metrics.precision_recall_curve(np.bitwise_not(self.close_samples), -scores)
        aupr_out = metrics.auc(recall,precision)

        det_acc = .5 * (tpr + 1.-fpr).max()

        if thresh < -99999:
            tidx = np.abs(np.array(tpr) - 0.95).argmin()
            thresh = thresholds[tidx]
        predicts = scores >= thresh
        ys = self.close_samples
        accuracy = metrics.accuracy_score(ys,predicts)
        f1 = metrics.f1_score(ys,predicts)
        recall = metrics.recall_score(ys,predicts)
        precision = metrics.precision_score(ys,predicts)
        fpr_at_tpr95 = fpr[tidx]
        return {
            "auroc" : auroc,
            "auprIN" : aupr_in,
            "auprOUT" : aupr_out,
            "accuracy" : accuracy,
            'f1' : f1,
            'recall' : recall,
            'precision' : precision,
            "fpr@tpr95" : fpr_at_tpr95,
            "bestdetacc" : det_acc,
        }
    
    # modified from APRL
    def compute_oscr(self,pred, scores):
        unk_scores = scores[self.test_labels < 0]
        kn_cond = self.test_labels >= 0
        kn_ct = kn_cond.sum()
        unk_ct = scores.shape[0] - kn_ct
        kn_scores = scores[kn_cond]
        kn_correct_pred = pred[kn_cond] == self.test_labels[kn_cond]
        
        def fpr(thr):
            return (unk_scores > thr).sum() / unk_ct
        
        def ccr(thr):
            ac_cond = (kn_scores > thr) & (kn_correct_pred)
            return ac_cond.sum() / kn_ct

        sorted_scores = -np.sort(-scores)
        # Cutoffs are of prediction values
        
        CCR = [0]
        FPR = [0] 

        for s in sorted_scores:
            CCR.append(ccr(s))
            FPR.append(fpr(s))
        CCR += [1]
        FPR += [1]

        # Positions of ROC curve (FPR, TPR)
        ROC = sorted(zip(FPR, CCR), reverse=True)
        OSCR = 0
        # Compute AUROC Using Trapezoidal Rule
        for j in range(len(CCR)-1):
            h =   ROC[j][0] - ROC[j+1][0]
            w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

            OSCR = OSCR + h*w

        return OSCR

    # score > thresh 表示是一个开放样本
    def open_reco_indexes(self,scores,thresh,rawpredicts):
        if np.isnan(scores).any() or np.isinf(scores).any():
            return {}
        predicts = rawpredicts.copy()
        if thresh < -99999:
            fpr, tpr, thresholds  =  metrics.roc_curve(self.close_samples, scores) 
            thresh = thresholds[np.abs(np.array(tpr) - 0.95).argmin()]
        predicts[scores <= thresh] = -1
        ys = self.test_labels.copy()
        ys[ys < 0] = -1
        accuracy = metrics.accuracy_score(ys,predicts)
        macro_f1 = metrics.f1_score(ys,predicts,average='macro')
        weighted_f1 = metrics.f1_score(ys,predicts,average='weighted')
        macro_recall = metrics.recall_score(ys,predicts,average='macro')
        weighted_recall = metrics.recall_score(ys,predicts,average='weighted')
        macro_precision = metrics.precision_score(ys,predicts,average='macro')
        weighted_precision = metrics.precision_score(ys,predicts,average='weighted')
        oscr = self.compute_oscr(rawpredicts,scores)
        closeacc_withrej = np.sum((predicts == self.test_labels) & self.close_samples) / self.close_samples_ct

        clswise = {}
        tot = 0
        numclas = np.max(self.test_labels) + 1
        if numclas < 50:
            for c in range(numclas):
                cond = rawpredicts == c
                sbc = self.close_samples[cond]
                ck = sbc.sum()
                if ck == 0 or ck == sbc.shape[0]:
                    clswise[f'class{c}'] = 0.5
                else:
                    fpr, tpr, thresholds  =  metrics.roc_curve(sbc, scores[cond]) 
                    auroc = metrics.auc(fpr,tpr)
                    clswise[f'class{c}'] = auroc
                    tot += auroc * cond.sum()
                # print("AUROC for class",c,"is",auroc,", with sample number",np.sum(cond))
            clswise['mean'] = tot / rawpredicts.shape[0] 
        else:
            numclas = 'Too many to analyse'

        return {
            'closeacc_withrej':closeacc_withrej,
            'accuracy' : accuracy,
            'macro_f1' : macro_f1,
            'oscr' : oscr,
            'weighted_f1' : weighted_f1,
            'macro_recall' : macro_recall,
            'weighted_recall' : weighted_recall,
            'macro_precision' : macro_precision,
            'weighted_precision' : weighted_precision,
            'classwise_auc' : clswise
        }
    
    def openscore_distribution(self,scores,pred,savename):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use('Agg')
        percentiles = np.array([0.25,99.75])
        ptiles = np.percentile(scores, percentiles)
        
        bins = np.linspace(ptiles[0], ptiles[1], 80)
        plt.hist(scores[self.test_labels <= -1], bins=bins, facecolor="red", alpha=0.4,label='test unknown',density = True)
        
        plt.hist(scores[self.test_labels > -1], bins=bins, facecolor="green", alpha=0.4,label='test known',density = True)

        fs = 16
        plt.legend(fontsize = fs)
        plt.yticks([])
        plt.xticks(fontsize = fs - 2)
        plt.xlabel('Score Value',fontsize = fs)
        plt.tight_layout()
        savename = savename.replace("/",'#')
        plt.savefig("./test_figs/"+savename+".jpg")
        plt.cla()
    
    def known_unknown_confusion(self,scores,pred):
        numcls = np.max(self.test_labels) + 1
        numukn = -np.min(self.test_labels) 
        kvu_confusion = np.zeros([numcls,numukn])
        upk_confusion = np.zeros([numukn,numcls])
        for i in range(numcls):
            for j in range(numukn):
                uj = -j-1
                cond = (self.test_labels == i) | (self.test_labels == uj)
                sbsc = scores[cond]
                sbgt = self.test_labels[cond] >= 0
                fpr, tpr, thresholds  =  metrics.roc_curve(sbgt, sbsc) 
                auroc = metrics.auc(fpr,tpr)
                kvu_confusion[i,j] = auroc
        
        for j in range(numukn):
            uj = -j-1
            cond = self.test_labels == uj
            p = pred[cond]
            for i in range(numcls):
                upk_confusion[j,i] = (p == i).sum()
        return kvu_confusion,upk_confusion

    def openset_recognition_curve(self,scores,pred):
        unk_scores = scores[self.test_labels < 0]
        kn_cond = self.test_labels >= 0
        kn_ct = kn_cond.sum()
        unk_ct = scores.shape[0] - kn_ct
        kn_scores = scores[kn_cond]
        kn_correct_pred = pred[kn_cond] == self.test_labels[kn_cond]
        
        def fpr(thr):
            return (unk_scores > thr).sum() / unk_ct
        
        def ccr(thr):
            ac_cond = (kn_scores > thr) & (kn_correct_pred)
            return ac_cond.sum() / kn_ct

        results = []
        sorted_scores = -np.sort(-scores)
        for s in sorted_scores:
            results.append([fpr(s),ccr(s)])
        return np.array(results)
