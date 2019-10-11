# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from torch.optim import Adam

from model import Model
from net import Net
from path import MODEL_PATH
from vgg import vgg16_bn,vgg19_bn,vgg11_bn

# 数据获取辅助类
dataset = Dataset()

# 模型操作辅助类
model = Model(dataset)

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


def eval(model, x_test, y_test):
    net.eval()
    batch_eval = model.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        outputs = net(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len

#### Model Training Configs ####
# cnn = Net().to(device)
net = vgg19_bn().to(device)
optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

#### Training ####
best_accuracy = 0
for i in range(args.EPOCHS):
    net.train()
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)  # 读取数据

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.long().to(device)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_test = x_test.float().to(device)
    y_test = y_test.long().to(device)

    outputs = net(x_train)
    _, prediction = torch.max(outputs.data, 1)

    optimizer.zero_grad()

    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    # 若测试准确率高于当前最高准确率，则保存模型
    train_accuracy = eval(model, x_test, y_test)
    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(net, MODEL_PATH, overwrite=True)
        print("############### step %d, best accuracy %g" % (i, best_accuracy))

    print('['+str(i) + "/" + str(args.EPOCHS)+'] acc: '+str(train_accuracy))
