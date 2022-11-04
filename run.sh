
for dataset in cifar10 svhn tinyimagenet
do
    for s in a b c d e
    do
        python3 main.py --gpu 0 --ds ./exps/$dataset/spl_$s.json --config ./configs/linear/$dataset.json --save linear_$dataset_$s --method cssr --test_interval 2
        python3 main.py --gpu 0 --ds ./exps/$dataset/spl_$s.json --config ./configs/pcssr/$dataset.json --save pcssr_$dataset_$s --method cssr --test_interval 2
        python3 main.py --gpu 0 --ds ./exps/$dataset/spl_$s.json --config ./configs/rcssr/$dataset.json --save rcssr_$dataset_$s --method cssr --test_interval 2
    done
done

# imagenet
# python3 main.py --gpu 0 --ds ./exps/imagenet/vs_inaturalist.json --config ./configs/rcssr/imagenet.json --save imagenet1k_rcssr --method cssr_ft