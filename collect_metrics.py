import json

methods = ['linear','pcssr','rcssr']
datasets = ['cifar10','svhn','tinyimagenet']
splits = ['a','b','c','d', 'e']

def get_metric(file,metric,is_last = True):
    with open(file,'r') as f:
        hist = json.load(f)
    # last metric
    if is_last:
        res = hist[-1]
        for m in metric.split('.'):
            if m not in res:
                res = -1
                break
            res = res[m]
    # best metric
    else:
        res = 0
        for epoch in hist:
            for m in metric.split('.'):
                if not m in epoch:
                    epoch = -1
                    break
                epoch = epoch[m]
            res = max(res,epoch)
    return res

def generate_tables(use_last = True):
    for ds in datasets:
        print("\nDataset",ds,"Last Epoch" if use_last else "Best Epoch")
        print('method','average',*splits,sep='\t')
        for mth in methods:
            metrics = [get_metric(f'./save/{mth}_{ds}_{s}/hist.json','open_detection.auroc',use_last) for s in splits]
            # print(metrics)
            avg = sum(metrics) / len(metrics)
            metrics = [avg] + metrics
            metrics = list(map(lambda x:'%.04f' % x,metrics))
            print(mth,*metrics,sep='\t')

generate_tables(True)
generate_tables(False)