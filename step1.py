# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10
from data.miccai import MICCAI
import argparse, sys
import numpy as np
import torchvision.models as models
import torch.nn as nn
import pickle
from data.chaoyang import CHAOYANG

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--noise_rate', type=float, default=0.2)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--dataset', type=str, default='chaoyang')
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=15)
parser.add_argument('--gpu', type=int, default=0, help='gpu_id')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 96
learning_rate = args.lr

# load dataset

if args.dataset == 'digestpath':
    input_channel = 3
    num_classes = 2

    args.epoch_decay_start = 15
    args.n_epoch = 30

    train_dataset = MICCAI(root="/root/miccai",
                        json_name="train.json",
                        train=True,
                        transform=transforms.Compose(
                            [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                transforms.ToTensor()]),
                        noise_type=args.noise_type,
                        noise_rate=args.noise_rate
                        )




if args.dataset == 'chaoyang':
    input_channel = 3
    num_classes = 4
    args.noise_rate = 0.15
    args.epoch_decay_start = 30
    args.n_epoch = 80

    train_dataset = CHAOYANG(root="/root/chaoyang-data",
                        json_name="train.json",
                        train=True,
                        transform=transforms.Compose(
                            [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                transforms.ToTensor()])
                        )
                 


mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate / 4
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1






def record_history(index, output, target, recorder):
    # pdb.set_trace()
    pred = F.softmax(output, dim=1).cpu().data
    for i, ind in enumerate(index):
        recorder[ind].append(pred[i][target.cpu()[i]].numpy().tolist())

    return


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
def train(train_loader, epoch, model, optimizer, recorder):
    train_total = 0
    train_correct = 0
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        logits = model(images)
        record_history(indexes, logits, labels, recorder)
        prec, _ = accuracy(logits, labels, topk=(1, 1))
        train_total += 1
        train_correct += prec

        loss = torch.sum(F.cross_entropy(logits, labels))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                % (epoch + 1, args.n_epoch, i + 1, len(train_dataset) // batch_size, prec, loss.data,
                   ))

    train_acc = float(train_correct) / float(train_total)
    return train_acc


# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()  # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()


    acc = 100 * float(correct) / float(total)
    return acc


def main():
    # load dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    recorder = [[] for i in range(train_dataset.__len__())]
    # bulid model
    cnn = models.resnet34(pretrained=False)
    cnn.fc = nn.Linear(in_features=512, out_features=num_classes)
    cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    epoch = 0


    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn.train()
        adjust_learning_rate(optimizer, epoch)

        train_acc= train(train_loader, epoch, cnn, optimizer,recorder)

        print('Epoch [%d/%d] train Accuracy on the %s train images: Model %.4f %% ' % (
        epoch + 1, args.n_epoch, len(train_dataset), train_acc))


        

    print(args)
    with open("record/%s_%d_iter1_1.p"%(args.dataset,int(args.noise_rate*100)), 'wb') as recordf:
        pickle.dump(recorder, recordf)


    #step2:select easy samples, add noise to D_e
    weight_record = np.array(recorder)[:,10:]
    np_train = np.array([np.mean(weight_record[i]) for i in range(len(weight_record))])
    index = np.argsort(np_train)
    filter_ratio = args.noise_rate * 1.5
    clean = index[int(len(index) * filter_ratio):]
    path_li = np.array(train_dataset.train_data)[clean].tolist()
    label_li = np.array(train_dataset.train_noisy_labels)[clean].tolist()
    # create D_a
    clean_dataset = MICCAI(root="",
                           path_list=path_li,
                           label_list=label_li,
                           train=True,
                           transform=transforms.Compose(
                               [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                transforms.ToTensor()]),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate,
                           nb_classes = num_classes
                           )
    clean_loader = torch.utils.data.DataLoader(dataset=clean_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    recorder3 = [[] for i in range(clean_dataset.__len__())] # H_a
    cnn = models.resnet34(pretrained=False)
    cnn.fc = nn.Linear(in_features=512, out_features=num_classes)
    cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    for epoch in range(1, args.n_epoch):
        # train models
        cnn.train()
        adjust_learning_rate(optimizer, epoch)
        train_acc= train(clean_loader, epoch, cnn, optimizer,recorder3) 
        print('Epoch [%d/%d] train Accuracy on the %s train images: Model %.4f %% ' % (
        epoch + 1, args.n_epoch, len(train_dataset), train_acc))

    with open("record/%s_%d_iter2_1.p"%(args.dataset,int(args.noise_rate*100)), 'wb') as recordf1:
        pickle.dump(recorder3, recordf1)


    # train EHN model
    train_history = []
    for j in range(clean_dataset.__len__()):
        if not clean_dataset.noise_or_not[j]:
            result = 1  # noise
        else:
            result = 0  # hard
        train_history.append({"record": recorder3[j], "label": result})
    record_train = [train_history[i]['record'] for i in range(len(train_history))]
    label_train = [int(train_history[i]['label']) for i in range(len(train_history))]
    record_train = torch.Tensor(record_train).cuda()
    label_train = torch.Tensor(label_train).type(torch.LongTensor).cuda()

    class MLP(torch.nn.Module):  
        def __init__(self, n_feature, n_hidden, n_output):
            super(MLP, self).__init__()  
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  
            self.out = torch.nn.Linear(n_hidden, n_output)  

        def forward(self, x):
            
            x = F.relu(self.hidden(x))  
            x = self.out(x)  
            return x

    net = MLP(n_feature=record_train.shape[1], n_hidden=200, n_output=2)  
    net = net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()
    max_epoch = 500
    for epoch in range(max_epoch):
        net.train()
        loss_sigma = 0.0  #
        correct = 0.0
        total = 0.0
        out = net(record_train)  

        loss = loss_func(out, label_train)  

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        _, predicted = torch.max(out.data, 1)
        total += label_train.size(0)
        correct += (predicted == label_train).squeeze().sum().cpu().numpy()
        loss_sigma += loss.item()
        print("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, max_epoch, loss_sigma, correct / total))

    torch.save(net.state_dict(), 'model/%s_%d_EHN_net.pth'%(args.dataset,int(args.noise_rate*100)))

    #sample correction and selection
    correction_model = models.resnet34(pretrained=False)
    correction_model.fc = nn.Linear(in_features=512, out_features=num_classes)
    correction_model.cuda()

    dirty = index[:int(len(index) * filter_ratio)]
    dirty_path_li = np.array(train_dataset.train_data)[dirty].tolist()
    dirty_label_li = np.array(train_dataset.train_noisy_labels)[dirty].tolist()
    dirty_dataset = MICCAI(root="",
                           path_list=dirty_path_li,
                           label_list=dirty_label_li,
                           train=False,
                           transform=transforms.Compose(
                               [transforms.Resize((256, 256)),
                                transforms.ToTensor()])
                           )
    dirty_loader = torch.utils.data.DataLoader(dataset=dirty_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=False)
    
    # classify hard and noisy samples
    record_test = torch.Tensor(recorder)[dirty].cuda()
    net.eval()
    out = net(record_test)
    out.detach_()
    _, predicted = torch.max(out.data, 1)
    predicted = predicted.tolist()
    noisy_list = []
    for i in range(dirty_dataset.__len__()):
        if predicted[i] == 1:
            noisy_list.append(i)
    e_h_index = list(set(index.tolist()) - set(dirty[noisy_list]))
   
    e_h_dataset = MICCAI(root="",
                           path_list=np.array(train_dataset.train_data)[e_h_index].tolist(),
                           label_list=np.array(train_dataset.train_noisy_labels)[e_h_index].tolist(),
                           train=False,
                           transform=transforms.Compose(
                               [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                transforms.ToTensor()])
                           )
    e_h_loader = torch.utils.data.DataLoader(dataset=e_h_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    recorder4 = [[] for i in range(e_h_dataset.__len__())]#实际上没用的，随便存存
    optimizer = torch.optim.Adam(correction_model.parameters(), lr=learning_rate)

    for epoch in range(1, args.n_epoch):
        # train correction models
        correction_model.train()
        adjust_learning_rate(optimizer, epoch)
        train_acc= train(e_h_loader, epoch, correction_model, optimizer,recorder4)
    
        print('Epoch [%d/%d] train Accuracy on the %s train images: Model %.4f %% ' % (
        epoch + 1, args.n_epoch, len(train_dataset), train_acc))

    correction_model.eval()

    # generate pseudo-labels
    pre = torch.zeros(dirty_dataset.__len__())
    p = torch.zeros(dirty_dataset.__len__())
    for images, labels, indexs in dirty_loader:
        images = Variable(images).cuda()
        logits1 = correction_model(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        pre[indexs] = pred1.cpu().float()
        p[indexs] = _.cpu().float()
    p_index = torch.argsort(p)
    pre = pre.tolist()
    
    # post-processing
    unconfirm = []
    unconfirm0 = []
    unconfirm1 = []
    for i in range(dirty_dataset.__len__()):
        if pre[i] != dirty_dataset.test_labels[i] and predicted[i] == 0: #纠正模型更改了标签，但EHN预测为hard
            unconfirm.append(i)
            unconfirm0.append(i)
        elif pre[i] == dirty_dataset.test_labels[i] and predicted[i] == 1: #纠正模型未更改标签，但EHN预测为noisy
            unconfirm.append(i)
            unconfirm1.append(i)
    # label correction 
    for i in range(dirty_dataset.__len__()):
        train_dataset.train_noisy_labels[dirty[i]] = np.int64(pre[i])
    # sample selection
    # new_clean = list(set(index.tolist()) - set(dirty[p_index[:int(dirty_dataset.__len__()*args.noise_rate)]].tolist()) - set(dirty[unconfirm]))
    new_clean = list(set(index.tolist()) - set(dirty[unconfirm]))
    num = 0
    for idx in new_clean:
        if train_dataset.train_noisy_labels[idx] != train_dataset.train_labels[idx]:
            num+=1

    print("new_clean_len:",len(new_clean))

    train_dataset.train_data = np.array(train_dataset.train_data)[new_clean].tolist()
    train_dataset.train_noisy_labels = np.array(train_dataset.train_noisy_labels)[new_clean].tolist()
    train_dataset.noise_or_not = np.array(train_dataset.noise_or_not)[new_clean]
    train_dataset.train_labels = np.array(train_dataset.train_labels)[new_clean]
    with open("%s_%d_step1.p"%(args.dataset,int(args.noise_rate*100)), 'wb') as f:
        pickle.dump(train_dataset, f)



if __name__ == '__main__':
    main()
