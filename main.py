#-*- coding: utf-8 -*-

import os
import datetime
import random
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from parse import parse_args
from torch import sparse
from tqdm import tqdm
from data import *
from model import *
from evaluation import *
from cuda import *
# print(torch.__version__)
USE_CUDA = torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if USE_CUDA else 'cpu')
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_path():
    directory = 'data/'
    if arg.dataset == '100k':
        total_file = directory + '/' + '100k.csv'
        train_file = directory + '/' + '100k_train.csv'
        test_file = directory + '/' + '100k_test.csv'
    elif arg.dataset == 'yahoo':
        total_file = directory + '/' + 'yahoo1.csv'
        train_file = directory + '/' + 'yahoo1_train.csv'
        test_file = directory + '/' + 'yahoo1_test.csv'
    elif arg.dataset == '1M':
        total_file = directory + '/' + '1m1.csv'
        train_file = directory + '/' + '1m1_train.csv'
        test_file = directory + '/' + '1m1_test.csv'
    elif arg.dataset == 'gowalla':
        total_file = directory + '/' + 'gowalla.csv'
        train_file = directory + '/' + 'gowalla_train.csv'
        test_file = directory + '/' + 'gowalla_test.csv'
    elif arg.dataset == 'amazon-book':
        total_file = directory + '/' + 'amazon-book.csv'
        train_file = directory + '/' + 'amazon-book_train.csv'
        test_file = directory + '/' + 'amazon-book_test.csv'
    elif arg.dataset == 'yelp2018':
        total_file = directory + '/' + 'yelp2018.csv'
        train_file = directory + '/' + 'yelp2018_train.csv'
        test_file = directory + '/' + 'yelp2018_test.csv'
    return total_file, train_file, test_file


def log():
    if arg.log:
        path = arg.log_root
        #path = arg.log_root + arg.dataset
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + '/' + arg.dataset + '_' + arg.LOSS + '_'+ arg.model + '_' + str(arg.M) + '_' + str(arg.N) + '_' +  '--' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
        f = open(file, 'w')
        print('----------------loging----------------')
    else:
        f = sys.stdout
    return f


def get_numbers_of_ui_and_divider(file):
    '''
    :param file: data path
    :return:
    num_users: total number of users
    num_items: total number of items
    '''
    data = pd.read_csv(file, header=0, dtype='str', sep=',')
    userlist = list(data['user'].unique())
    itemlist = list(data['item'].unique())
    popularity = np.zeros(len(itemlist))
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
    num_users, num_items = len(userlist), len(itemlist)
    return num_users, num_items


def load_train_data(path, num_item):
    data = pd.read_csv(path, header=0, sep=',')
    datapair = []
    popularity = np.zeros(num_item)
    train_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
        datapair.append((user, item))
        train_tensor[user, item] = 1
    return train_tensor.to_sparse(), datapair


def load_test_data(path):
    data = pd.read_csv(path, header=0, sep=',')
    test_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        test_tensor[user, item] = 1
    return test_tensor.bool()

def model_init():
    # A new train
    model_path = r'.\model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if arg.train_mode == 'new_train':
        if arg.model == 'OneBP':
            model = OneBP(num_users, num_items, arg, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_dc_epoch, gamma=arg.lr_dc)
        checkpoint = 0
    # Continue train
    else:
        checkpoint = torch.load(r'.\model\{}-{}-{}-{}-ex_model.pth'.format(arg.dataset, arg.model, arg.LOSS, arg.beta))
        if arg.model == 'OneBP':
            model = OneBP(num_users, num_items, arg, device)
        model.load_state_dict(checkpoint['net'])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_dc_epoch, gamma=arg.lr_dc)
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('epoch_begin:', checkpoint['epoch'] + 1)
    return model, optimizer, scheduler, checkpoint


def model_train(real_epoch):
    print('-------------------------------------------', file=f)
    print('-------------------------------------------')
    print('epoch: ', real_epoch, file=f)
    print('epoch: ', real_epoch)
    print('start training: ', datetime.datetime.now(), file=f)
    print('start training: ', datetime.datetime.now())
    st = time.time()
    model.train()
    total_loss = []

    for index, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        # To device
        batch = batch.to(device)

        # Fetch Data
        users = batch[:, 0]  # [bs,]
        positives = batch[:, 1:arg.M+1]    # [bs * M]
        negatives = batch[:, arg.M+1:]     # [bs * N]

        # Calculate Loss
        loss = model(users, positives, negatives, real_epoch, optimizer)

        total_loss.append(loss.item())

    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']), file=f)
    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Training time:[%0.2f s]' % (time.time() - st))
    print('Training time:[%0.2f s]' % (time.time() - st), file=f)


def model_test():
    print('----------------', file=f)
    print('----------------')
    print('start evaluation: ', datetime.datetime.now(), file=f)
    print('start evaluation: ', datetime.datetime.now())
    model.eval()

    Pre_dic, Recall_dict, F1_dict, NDCG_dict = {}, {}, {}, {}
    sp = time.time()
    rating_mat = model.predict() # |U| * |V|
    rating_mat = erase(rating_mat)
    for k in arg.topk:
        metrices = topk_eval(rating_mat, k, test_tensor.to(device))
        precision, recall, F1, ndcg = metrices[0], metrices[1], metrices[2], metrices[3]
        Pre_dic[k] = precision
        Recall_dict[k] = recall
        F1_dict[k] = F1
        NDCG_dict[k] = ndcg
    print('Evaluation time:[%0.2f s]' % (time.time() - sp))
    print('Evaluation time:[%0.2f s]' % (time.time() - sp), file=f)
    return Pre_dic, Recall_dict, F1_dict, NDCG_dict


def erase(score):
    x = train_tensor.to(device) * (-1000)
    score = score + x
    return score


def print_epoch_result(real_epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict):

    for k in arg.topk:
        if Pre_dic[k] > best_result[k][0]:
            best_result[k][0], best_epoch[k][0] = Pre_dic[k], real_epoch
        if Recall_dict[k] > best_result[k][1]:
            best_result[k][1], best_epoch[k][1] = Recall_dict[k], real_epoch
        if F1_dict[k] > best_result[k][2]:
            best_result[k][2], best_epoch[k][2] = F1_dict[k], real_epoch
        if NDCG_dict[k] > best_result[k][3]:
            best_result[k][3], best_epoch[k][3] = NDCG_dict[k], real_epoch
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k]))
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k]),
            file=f)
    return best_result, best_epoch


def print_best_result(best_result, best_epoch):
    print('------------------best result-------------------', file=f)
    print('------------------best result-------------------')
    for k in arg.topk:
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3],  (time.time() - t0)))
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3],  (time.time() - t0)), file=f)

        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3],
                (time.time() - t0)))
        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3],
                (time.time() - t0)), file=f)
    print('------------------------------------------------', file=f)
    print('------------------------------------------------')
    print('Run time: %0.2f s' % (time.time() - t0), file=f)
    print('Run time: %0.2f s' % (time.time() - t0))


if __name__ == '__main__':
    t0 = time.time()
    arg = parse_args()
    print(arg)
    f = log()
    print(arg,file=f)
    init_seed(2022)
    total_file, train_file, test_file = get_data_path()
    num_users, num_items = get_numbers_of_ui_and_divider(total_file)

    # Load Data
    train_tensor, train_pair = load_train_data(train_file, num_items)
    test_tensor = load_test_data(test_file)

    dataset = Data(train_pair, arg, num_users, num_items)
    train_loader = DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True, pin_memory=True, num_workers=arg.num_workers)

    # Init Model
    model, optimizer, scheduler, checkpoint = model_init()
    best_result = {}
    best_epoch = {}
    for k in arg.topk:
        best_result[k] = [0., 0., 0., 0.]
        best_epoch[k] = [0, 0, 0, 0]

    # Train and Test
    for epoch in range(arg.epochs):
        if arg.train_mode == 'new_train':
            real_epoch = epoch
        else:
            real_epoch = checkpoint['epoch'] + 1 + epoch
        model_train(real_epoch)
        # mc.show_cuda_info()
        Pre_dic, Recall_dict, F1_dict, NDCG_dict = model_test()

        scheduler.step()
        best_result, best_epoch = print_epoch_result(real_epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict)
    print_best_result(best_result, best_epoch)
    f.close()

    # Save Checkpoint
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': real_epoch}
    torch.save(state, r'.\model\{}-{}-{}-{}-ex_model.pth'.format(arg.dataset, arg.model, arg.LOSS, arg.beta))






