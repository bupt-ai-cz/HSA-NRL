# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.miccai import MICCAI
import argparse, sys
import numpy as np
import torchvision.models as models
import torch.nn as nn
import pickle


from loss import loss_weight,loss_noweight


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 1e-3)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--dataset', type = str, help = 'digestpath', default = 'digestpath')
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=18)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--warm_up', type=int, default=10)
parser.add_argument('--pickle_path', type=str, required=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters

learning_rate = args.lr 

# load dataset


if args.dataset=='digestpath':

    input_channel=3
    num_classes=2

    args.epoch_decay_start = 15
    args.n_epoch = 40
    batch_size = 96

    train_dataset = pickle.load(open(args.pickle_path,"rb"))

    test_dataset = MICCAI(root="/root/miccai",
                          json_name="test.json",
                          train=False,
                          transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
                          )
if args.dataset=='camelyon':

    input_channel=3
    num_classes=2
    args.epoch_decay_start = 15
    args.n_epoch = 40
    batch_size = 96

    train_dataset = pickle.load(open(args.pickle_path,"rb"))
    print(train_dataset.__len__())

    test_dataset = MICCAI(root="/root/camelyon16_patch",
                          json_name="test.json",
                          train=False,
                          transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
                          )

if args.dataset=='chaoyang':

    input_channel=3
    num_classes=4
    args.epoch_decay_start = 30
    args.n_epoch = 80
    batch_size = 96
    train_dataset = pickle.load(open(args.pickle_path,"rb"))

    test_dataset = MICCAI(root="/root/chaoyang-data",
                          json_name="test_new.json",
                          train=False,
                          transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
                          )

recorder1 = [[] for i in range(train_dataset.__len__())]
recorder2 = [[] for i in range(train_dataset.__len__())]
def record_history(index,output,target,recorder):
    # pdb.set_trace()
    pred = F.softmax(output, dim=1).cpu().data
    # pred = output.cpu().data
    # _, pred = torch.max(F.softmax(output, dim=1).data, 1)
    for i,ind in enumerate(index):
        recorder[ind].append(pred[i][target.cpu()[i]].numpy().tolist())
        ##save forget event below
        # recorder[ind].append((target.cpu()[i] == pred.cpu()[i]).numpy().tolist())
    return


# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, m):
    print ('Training ...' )

    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 
    if epoch != 1:
        m_prob_1 = np.array([1-(recorder1[i][-1]) for i in range(len(recorder1))])
        m_prob_2 = np.array([1-(recorder2[i][-1]) for i in range(len(recorder2))])
        m_prob_1 = torch.from_numpy(m_prob_1).cuda().float()
        m_prob_2 = torch.from_numpy(m_prob_2).cuda().float()
        m_prob_1_sorted_index = torch.argsort(m_prob_1)
        m_prob_2_sorted_index = torch.argsort(m_prob_2)
        forget_threshold = int(args.forget_rate*len(recorder1))
        if forget_threshold == 0:
            drop_ind1 = torch.tensor([])
            drop_ind2 = torch.tensor([])
        else:
            drop_ind1 = m_prob_1_sorted_index[-forget_threshold:]
            drop_ind2 = m_prob_2_sorted_index[-forget_threshold:]
    else:#  recorder is empty
        drop_ind1 = torch.tensor([])
        drop_ind2 = torch.tensor([])

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        logits1=model1(images)
        record_history(indexes,logits1,labels,recorder1)
        prec1, _ = accuracy(logits1, labels, topk=(1, 1))
        train_total+=1
        train_correct+=prec1
        with torch.no_grad():
            logits2 = model2(images)
        record_history(indexes,logits2,labels,recorder2)
        prec2, _ = accuracy(logits2, labels, topk=(1, 1))
        train_total2+=1
        train_correct2+=prec2
        if epoch < args.warm_up:# warm up
            loss_1, loss_2 = loss_noweight(logits1, logits2, labels, ind, drop_ind1, drop_ind2)
        else:
            loss_1, loss_2 = loss_weight(logits1, logits2, labels, ind, recorder1, recorder2, drop_ind1, drop_ind2)


        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        with torch.no_grad():
            for param_1, param_2 in zip(model1.parameters(), model2.parameters()):
                param_2.data = param_2.data * m + param_1.data * (1. - m)

        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.data, loss_2.data, ))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print ('Evaluating ...')
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def main():
    # Data Loader (Input Pipeline)
    print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    # Define models
    print ('building model...')
    cnn1 = models.resnet34(pretrained=False)
    cnn1.fc = nn.Linear(in_features=512, out_features=num_classes)

    cnn1.cuda()
    #print (cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)

    cnn2 = models.resnet34(pretrained=False)
    cnn2.fc = nn.Linear(in_features=512, out_features=num_classes)
    cnn2.cuda()
    with torch.no_grad():
        for param_1, param_2 in zip(cnn1.parameters(), cnn2.parameters()):
            param_2.data.copy_(param_1.data)  # initialize
            param_2.requires_grad = False  # not update by gradient
    #print (cnn2.parameters)
    epoch=0
    best_acc = 0
    # training
    for epoch in range(1, args.n_epoch):
        # train models

        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        train_acc1, train_acc2=train(train_loader, epoch, cnn1, optimizer1, cnn2, m=0.999)

        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        if test_acc1 > best_acc:
            best_acc = test_acc1
        if test_acc2 > best_acc:
            best_acc = test_acc2
        print('Epoch [%d/%d] test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    print(best_acc)
    print(args)
    
if __name__=='__main__':
    main()
