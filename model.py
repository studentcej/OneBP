import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
# OneBP
def criterion(pos_scores, neg_scores,arg, epoch):
    '''
    :param pos_scores: bs * M
    :param neg_scores: bs * N
    :return: loss
    '''
    if arg.LOSS == 'Info_NCE':
        loss = (- torch.log(pos_scores[:, 0] / (pos_scores[:, 0] + neg_scores.sum(dim=-1)))).mean()
    else:
        print('Invalid loss function')
        raise Exception
    return loss

class OneBP(nn.Module):
    def __init__(self, num_users, num_items, arg, device):
        super(OneBP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.arg = arg
        self.device = device

        self.dim = arg.dim

        self.User_Emb = nn.Embedding(self.num_users, self.dim)
        nn.init.xavier_normal_(self.User_Emb.weight)
        self.Item_Emb = nn.Embedding(self.num_items, self.dim)
        nn.init.xavier_normal_(self.Item_Emb.weight)

        # Cut off the gradient of User Tower
        self.User_Emb.requires_grad = False

    def computer(self):
        users_emb = self.User_Emb.weight
        items_emb = self.Item_Emb.weight
        return users_emb, items_emb


    @torch.no_grad()
    def update(self, users, positives):
        # Moving-aggregation Update
        self.User_Emb.weight.data[users] = self.arg.beta * self.User_Emb.weight.data[users] + (1. - self.arg.beta) * self.Item_Emb.weight.data[positives]


    def forward(self, users, positives, negatives, epoch, optimizer):

        all_users_emb, all_items_emb = self.computer()

        pos_item_embs = all_items_emb[positives]  # bs * M * d
        neg_item_embs = all_items_emb[negatives]  # bs * N * d

        with torch.no_grad():
            users_emb = all_users_emb[users]  # bs * d
            users_emb.detach()

        # Calculate Pos scores
        # [bs * 1 * d] * [bs * M * d]
        pos_scores = (users_emb.unsqueeze(1) * pos_item_embs).sum(dim=-1)  # bs * M
        pos_scores = torch.exp(pos_scores / self.arg.temperature)
        # Calculate Neg
        neg_scores = (users_emb.unsqueeze(1) * neg_item_embs).sum(dim=-1)  # bs * N
        neg_scores = torch.exp(neg_scores / self.arg.temperature)

        loss = criterion(pos_scores,neg_scores,self.arg, epoch)
        loss.backward()
        # Item tower Update
        optimizer.step()

        # User tower Update
        self.update(users, positives[:, 0])
        return loss

    def predict(self):
        all_users_emb = self.User_Emb.weight
        all_items_emb = self.Item_Emb.weight
        rate_mat = torch.mm(all_users_emb, all_items_emb.t())
        return rate_mat

    def calculate_score(self, users):
        all_users_emb = self.User_Emb.weight
        all_items_emb = self.Item_Emb.weight
        users_emb = all_users_emb[users]
        rate_score = torch.mm(users_emb, all_items_emb.t())
        return rate_score