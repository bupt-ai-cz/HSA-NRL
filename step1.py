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
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')


parser.add_argument('--dataset', type=str, help=' cifar10, miccai', default='miccai')
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=16, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=18)
parser.add_argument('--gpu', type=int, default=0, help='gpu_id')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 256
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


    test_dataset = MICCAI(root="/root/miccai",
                          json_name="test.json",
                          train=False,
                          transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
                          )

if args.dataset == 'camelyon':
    input_channel = 3
    num_classes = 2

    args.epoch_decay_start = 15
    args.n_epoch = 30

    train_dataset = MICCAI(root="/root/camelyon16_patch",
                        json_name="train.json",
                        train=True,
                        transform=transforms.Compose(
                            [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                transforms.ToTensor()]),
                        noise_type=args.noise_type,
                        noise_rate=args.noise_rate
                        )


    test_dataset = MICCAI(root="/root/camelyon16_patch",
                          json_name="test.json",
                          train=False,
                          transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
                          )
if args.dataset == 'chaoyang':
    input_channel = 3
    num_classes = 4

    args.epoch_decay_start = 15
    args.n_epoch = 30

    train_dataset = CHAOYANG(root="/root/chaoyang-data",
                        json_name="train_new.json",
                        train=True,
                        transform=transforms.Compose(
                            [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                transforms.ToTensor()]),
                        noise_type="clean",
                        noise_rate=args.noise_rate
                        )


    test_dataset = CHAOYANG(root="/root/chaoyang-data",
                          json_name="test_new.json",
                          train=False,
                          transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
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
    # pred = output.cpu().data
    # _, pred = torch.max(F.softmax(output, dim=1).data, 1)
    for i, ind in enumerate(index):
        recorder[ind].append(pred[i][target.cpu()[i]].numpy().tolist())
        ##save forget event below
        # recorder[ind].append((target.cpu()[i] == pred.cpu()[i]).numpy().tolist())
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


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    recorder = [[] for i in range(train_dataset.__len__())]
    # bulid model
    cnn = models.resnet34(pretrained=False)
    cnn.fc = nn.Linear(in_features=512, out_features=num_classes)
    cnn.cuda()
    # print (cnn1.parameters)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    best_acc = 0
    epoch = 0
    train_acc = 0

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn.train()
        adjust_learning_rate(optimizer, epoch)

        train_acc= train(train_loader, epoch, cnn, optimizer,recorder)

        test_acc= evaluate(test_loader, cnn)


        print('Epoch [%d/%d] test Accuracy on the %s test images: Model %.4f %% ' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc))

    torch.save(cnn.state_dict(), 'model/%s_%d_model_iter1.pth'%(args.dataset,int(args.noise_rate*100)))

        

    print(args)
    with open("record/%s_%d_iter1_1.p"%(args.dataset,int(args.noise_rate*100)), 'wb') as recordf:
        pickle.dump(recorder, recordf)


    #step2:挑选easy_dataset，得到训练区分hard和noise样本的数据
    weight_record = np.array(recorder)[:,20:]
    np_train = np.array([np.mean(weight_record[i]) for i in range(len(weight_record))])
    index = np.argsort(np_train)
    filter_ratio = args.noise_rate * 1.5
    clean = index[int(len(index) * filter_ratio):]
    path_li = np.array(train_dataset.train_data)[clean].tolist()
    label_li = np.array(train_dataset.train_noisy_labels)[clean].tolist()
    clean_dataset = MICCAI(root="",
                           path_list=path_li,
                           label_list=label_li,
                           train=True,
                           transform=transforms.Compose(
                               [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                transforms.ToTensor()]),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate
                           )
    clean_loader = torch.utils.data.DataLoader(dataset=clean_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    recorder3 = [[] for i in range(clean_dataset.__len__())]
    cnn = models.resnet34(pretrained=False)
    cnn.fc = nn.Linear(in_features=512, out_features=num_classes)
    cnn.cuda()
    # print (cnn1.parameters)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)


    for epoch in range(1, args.n_epoch):
        # train models
        cnn.train()
        adjust_learning_rate(optimizer, epoch)

        train_acc= train(clean_loader, epoch, cnn, optimizer,recorder3)

        test_acc= evaluate(test_loader, cnn)

        
        print('Epoch [%d/%d] test Accuracy on the %s test images: Model %.4f %% ' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc))

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

    class MLP(torch.nn.Module):  # 继承 torch 的 Module
        def __init__(self, n_feature, n_hidden, n_output):
            super(MLP, self).__init__()  # 继承 __init__ 功能
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
            self.out = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

        def forward(self, x):
            # 正向传播输入值, 神经网络分析出输出值
            x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
            x = self.out(x)  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
            return x

    net = MLP(n_feature=record_train.shape[1], n_hidden=200, n_output=2)  # 几个类别就几个 output
    net = net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()
    max_epoch = 500
    for epoch in range(max_epoch):
        net.train()
        loss_sigma = 0.0  #
        correct = 0.0
        total = 0.0
        out = net(record_train)  # 喂给 net 训练数据 x, 输出分析值

        loss = loss_func(out, label_train)  # 计算两者的误差

        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters

        _, predicted = torch.max(out.data, 1)
        total += label_train.size(0)
        correct += (predicted == label_train).squeeze().sum().cpu().numpy()
        loss_sigma += loss.item()
        print("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, max_epoch, loss_sigma, correct / total))


    torch.save(net.state_dict(), 'model/%s_%d_CHN_net.pth'%(args.dataset,int(args.noise_rate*100)))

    #sample correction and selection
    correction_model = models.resnet34(pretrained=False)
    correction_model.fc = nn.Linear(in_features=512, out_features=num_classes)
    correction_model.load_state_dict(torch.load("model/%s_%d_model_iter1.pth"%(args.dataset,int(args.noise_rate*100))))
    correction_model.cuda()
    correction_model.eval()  # Change model to 'eval' mode.
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
    pre = torch.zeros(dirty_dataset.__len__())
    p = torch.zeros(dirty_dataset.__len__())
    #预测噪声标签的新label
    for images, labels, indexs in dirty_loader:
        images = Variable(images).cuda()
        logits1 = correction_model(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        pre[indexs] = pred1.cpu().float()
        p[indexs] = _.cpu().float()
    p_index = torch.argsort(p)
    pre = pre.tolist()
    #根据历史预测是否为噪声标签
    record_test = torch.Tensor(recorder)[dirty].cuda()
    net.eval()
    out = net(record_test)
    out.detach_()
    _, predicted = torch.max(out.data, 1)
    predicted = predicted.tolist()
    unconfirm = []
    unconfirm0 = []
    unconfirm1 = []
    for i in range(dirty_dataset.__len__()):
        if pre[i] != dirty_dataset.test_labels[i] and predicted[i] == 0: #纠正模型更改了标签，但EHN预测为hard
            unconfirm.append(i)
            unconfirm0.append(i)
        elif pre[i] == dirty_dataset.test_labels[i] and predicted[i] == 1: #纠正模型未更改标签，但EHN预测为noise
            unconfirm.append(i)
            unconfirm1.append(i)
    #标签纠正
    for i in range(dirty_dataset.__len__()):
        train_dataset.train_noisy_labels[dirty[i]] = np.int64(pre[i])
    #标签筛选
    # new_clean = list(set(index.tolist()) - set(dirty[p_index[:int(dirty_dataset.__len__()*args.noise_rate)]].tolist()) - set(dirty[unconfirm]))
    new_clean = list(set(index.tolist()) - set(dirty[unconfirm]))
    num = 0
    for idx in new_clean:
        if train_dataset.train_noisy_labels[idx] != train_dataset.train_labels[idx]:
            num+=1
    print("new_clean_len:",len(new_clean))
    print("noise_rate:",num/len(new_clean))
    num = 0
    for idx in range(train_dataset.__len__()):
        if train_dataset.train_noisy_labels[idx] != train_dataset.train_labels[idx]:
            num+=1
    print("noise_rate2:",num/train_dataset.__len__())
    with open("%s_%d_step1_filtered_dataset_iter1.p"%(args.dataset,int(args.noise_rate*100)), 'wb') as f:
        pickle.dump(train_dataset, f)
    train_dataset.train_data = np.array(train_dataset.train_data)[new_clean].tolist()
    train_dataset.train_noisy_labels = np.array(train_dataset.train_noisy_labels)[new_clean].tolist()
    train_dataset.noise_or_not = np.array(train_dataset.noise_or_not)[new_clean]
    train_dataset.train_labels = np.array(train_dataset.train_labels)[new_clean]
    with open("%s_%d_step1_filtered_dataset_iter1_over.p"%(args.dataset,int(args.noise_rate*100)), 'wb') as f:
        pickle.dump(train_dataset, f)

    #step2

if __name__ == '__main__':
    main()
