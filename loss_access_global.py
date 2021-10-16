import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb

# Loss functions
def loss_access(y_1, y_2, t, ind, record_1, record_2, drop_idx1, drop_idx2):
    ind = ind.tolist()
    drop_idx1 = drop_idx1.tolist()
    drop_idx2 = drop_idx2.tolist()
    forget_ind1 = list(set(ind) & set(drop_idx1))
    forget_ind2 = list(set(ind) & set(drop_idx2))

    ind1 = list(range(len(ind)))
    ind2 = list(range(len(ind)))
    for i in forget_ind1:
        ind1.remove(ind.index(i))
    for i in forget_ind2:
        ind2.remove(ind.index(i))


    '''
    m_prob_1 = np.array([1-(np.mean(record_1[ind1[i]])) for i in range(len(ind1))])
    m_prob_2 = np.array([1-(np.mean(record_2[ind2[i]])) for i in range(len(ind2))])
    '''
    m_prob_1 = np.array([1-(record_1[ind1[i]][-1]) for i in range(len(ind1))])
    m_prob_2 = np.array([1-(record_2[ind2[i]][-1]) for i in range(len(ind2))])

    m_prob_1 = torch.from_numpy(m_prob_1).cuda().float()
    m_prob_2 = torch.from_numpy(m_prob_2).cuda().float()
    weight_1 = torch.pow(1-m_prob_1,2)
    weight_2 = torch.pow(1-m_prob_2,2)

    loss_1_update = (F.cross_entropy(y_1[ind2], t[ind2])*weight_1)
    loss_2_update = (F.cross_entropy(y_2[ind1], t[ind1])*weight_2)

    return torch.sum(loss_1_update), torch.sum(loss_2_update)

def loss_noweight(y_1, y_2, t, ind, drop_idx1, drop_idx2):
    ind = ind.tolist()
    drop_idx1 = drop_idx1.tolist()
    drop_idx2 = drop_idx2.tolist()
    forget_ind1 = list(set(ind) & set(drop_idx1))
    forget_ind2 = list(set(ind) & set(drop_idx2))

    ind1 = list(range(len(ind)))
    ind2 = list(range(len(ind)))
    for i in forget_ind1:
        ind1.remove(ind.index(i))
    for i in forget_ind2:
        ind2.remove(ind.index(i))

    # exchange
    loss_1_update = (F.cross_entropy(y_1[ind2], t[ind2]))
    loss_2_update = (F.cross_entropy(y_2[ind1], t[ind1]))

    return torch.sum(loss_1_update), torch.sum(loss_2_update)
