## Brief Introduction
This thesis has proposed a new gradient backpropagation strategy for two tower recommendation models, called OneBP. The basic idea is to replacing the gradient backpropagation for the user encoding tower by our proposed a moving-aggregation to update user encodings.
## Prerequisites
- Python 3.8 
- PyTorch 1.11.0

## Some Tips
Flags in `parse.py`:

Model training related settings:

- `--train_mode` Choosing to either start a new training session, or continue training with the model saved from your previous session.
- `--epochs` Number of sweeps over the dataset to train.
- `--dataset` Choosing 100k, 1M, Gowalla, Yelp2018 or your dataset.

You can set the relevant parameters for model training,

- `--batch_size` size of each batch
- `--l2` l2 regulation constant.
- `--lr` learning rate.
- `--lr_dc` learning rate decay rate.
- `--lr_dc_epoch` training epoch at which the learning rate starts to decay.
- `--N` number of samples for negative sampling

#### Suggested Model Training Parameters
|                    | batch_size |  l2  |  lr  | lr_dc | lr_dc_epoch | dim  | 
|--------------------|:----------:|:----:|:----:|:-----:|:-----------:|------|
| 100k-MF            |    1024    | 1e-5 | 5e-4 |   1   |     []      | 32   |
| 1M-MF              |    1024    | 1e-6 | 1e-3 |   1   |     []      | 128  |
| Gowalla-MF         |    1024    |  0   | 5e-4 |   1   |     []      | 1024 |
| Yelp2018-MF        |    1024    |  0   | 5e-5 |   1   |     []      | 2048 |


OneBP related parameters:
- `--beta` modifies the degree of user aggregation to items

Suggested AUC_NS parameters are:
#### Suggested AUC_NS Parameters
|                   | $\beta$ |
|-------------------|:-------:|
| 100k-MF           |  0.99   |
| 1M-MF             |  0.99   |
| Gowalla-MF        |  0.999  |
| Yelp2018-MF       |  0.999  |

For instance, execute the following command to train CF model using AUC_NS method.
```
python main.py --model OneBP --dataset 100k --l2 1e-5 --lr 5e-4 --dim 32  --batch_size 1024 --beta 0.99 --N 5 --epochs 2000
python main.py --model OneBP --dataset 1M   --l2 1e-6 --lr 1e-3 --dim 128 --batch_size 1024 --beta 0.99 --N 5 --epochs 200
```


## Detailed infomations on Exploration Study with fixed category "genres"
Number of items in different categories:
![image-20240612110744041](C:\Users\69451\AppData\Roaming\Typora\typora-user-images\image-20240612110744041.png)

Since the label of each item in the 100k dataset is not unique, we judge how good the user embedded learning is by calculating the classification performance for each category, and the results of the experiment are shown below.

![class](E:\GitCode\OneBP\class.png)

Experiments demonstrate that OneBP achieves improved classification performance on most classes, and that the model takes more account of recommendations from other user interests (i.e., diversity)