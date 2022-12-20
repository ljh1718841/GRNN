import numpy as np
import time
import os
import random
from tqdm import tqdm
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from grnn import GRNN
from utils import load_data, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='cora', help='the name of dataset conluding cora, citeseer, pubmed')
parser.add_argument('--hidden_size', type=int, default=16, help='the hidden size of the GRNN model')
parser.add_argument('--lr', type=float, default=0.01, help='learing rate')
parser.add_argument('--activation', type=str, default='elu', help='activation function')
parser.add_argument('--l2_loss', type=float, default=5e-4, help='the regularization parameter')
parser.add_argument('--feat_drop', type=float, default=0.5, help='the dropout of features')
parser.add_argument('--attn_drop', type=float, default=0.5, help='the dropout of edges')
parser.add_argument('--input_drop', type=float, default=0.0, help='the dropout of nodes')
parser.add_argument('--num_heads', nargs='+', type=int, default=[8,1], help='the number of the head on each layer')
parser.add_argument('--residual', type=bool, default=False, help='the residual learning')
parser.add_argument('--Epoch', type=int, default=1000)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--iteration_times', type=int, default=3, help='the smoothing times')
parser.add_argument('--times', type=int, default=4, help='the times of running GRNN model')
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--save_path', type=str, default='GRNN_data', help='the path of saving data and model')
args = parser.parse_args()

name = args.name
hidden_size = args.hidden_size
lr = args.lr
activation = args.activation
l2_loss = args.l2_loss
feat_drop = args.feat_drop
attn_drop = args.attn_drop
input_drop = args.input_drop
num_heads = args.num_heads
residual = args.residual
Epoch = args.Epoch
patience = args.patience
iteration_times = args.iteration_times
times = args.times
use_gpu = args.use_gpu
save_path = args.save_path

graph, features, labels, num_classes, train_index, val_index, test_index = load_data(name)    
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path+'\\data\\'+name):
    os.makedirs(save_path+'\\data\\'+name)


def Upgrade(model, path, isOnehot = False):
    model.load_dict(paddle.load(path))
    model.eval()
    paddle.set_device('gpu')
    target = F.softmax(model(graph, features))

    if isOnehot == True:
        target = paddle.argmax(target, axis=-1)
        updates = paddle.gather(labels, train_index)
        target = paddle.scatter(target, train_index, updates)
        target = F.one_hot(target, len(set(labels.numpy())))
        target.stop_gradient = True
    elif isOnehot == False:
        updates = F.one_hot(paddle.gather(labels, train_index), len(set(labels.numpy())))
        paddle.scatter_(target, train_index, updates).stop_gradient = True #cover target
    return target

def Trainer(model, pathNet, mode='pre'):
    model.load_dict(paddle.load(pathNet))
    optimizer = paddle.optimizer.Adam(learning_rate=lr, weight_decay=l2_loss, parameters=model.parameters())
    if use_gpu:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')

    dur = []
    currect_step = 0
    val_acc_max = 0.0
    test_acc = 0.0
    test_acc_epoch = 0
    val_loss_min = np.inf
    for epoch in range(Epoch):
        if epoch >=3:
            t0 = time.time()

 
        masks = paddle.bernoulli(1. - paddle.ones([features.shape[0], 1]) * 0.5)
        X = masks * features
        model.train()
        logits = model(graph, X)
        
        
        if mode == 'pre':
            loss = F.cross_entropy(paddle.gather(logits, train_index), paddle.gather(labels, train_index)) 
        elif mode == 'upgrade':
            logits = model(graph, X)
            Y = F.cross_entropy(paddle.gather(logits, train_index), paddle.gather(labels, train_index))
            Z = F.cross_entropy(logits, target, soft_label=True)
            loss = paddle.exp(Y) + paddle.exp(Z) - 2
        elif mode == 'smooth':
            loss = F.cross_entropy(logits, target, soft_label=True)
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        if epoch >=3:
            dur.append(time.time() - t0)
        
        
        model.eval()
        logits = model(graph, features)
        train_acc = evaluate(logits, labels, train_index)
        val_acc = evaluate(logits, labels, val_index)
        acc = evaluate(logits, labels, test_index)
        if epoch % 10 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}"
                    .format(epoch, float(loss.numpy()), float(train_acc), float(val_acc), float(acc), np.mean(dur)))
        
        #Early stop
        val_loss = F.cross_entropy(paddle.gather(logits, val_index), paddle.gather(labels, val_index)).numpy()
        if ((val_acc >= val_acc_max or val_loss <= val_loss_min) and name=='cora') or \
            (val_acc >= val_acc_max and name=='citeseer') or \
                ((val_acc >= val_acc_max or val_loss <= val_loss_min) and name=='pubmed'):
            if (val_loss <= val_loss_min and name=='cora') or \
                (val_acc >= val_acc_max and name=='citeseer') or \
                    (val_acc >= val_acc_max and name=='pubmed'):
                        
                val_acc_early_model = val_acc
                val_loss_early_model = val_loss
                test_acc = acc
                test_acc_epoch = epoch
                path = save_path+'\\'+name+'_mode'
                paddle.save(model.state_dict(), path)
            val_acc_max = np.max((val_acc, val_acc_max))
            val_loss_min = np.min((val_loss, val_loss_min))
            currect_step = 0
        else:
            currect_step += 1
            if currect_step == patience:
                print('Early stop! Min loss: ', val_loss_min, ', Max accuracy: ', val_acc_max)
                print('Early stop model validation loss: ', val_loss_early_model, ', accuracy: ', val_acc_early_model)
                print('Early stop! Test accuracy:', test_acc, ', Epoch:', test_acc_epoch)
                break
    return path, val_loss_early_model[0], val_acc_early_model[0], test_acc[0]



# create model
net = GRNN(
        input_size = features.shape[1],
        hidden_size = hidden_size,
        feat_drop = feat_drop,
        attn_drop = attn_drop,
        input_drop= input_drop,
        num_class = len(set(labels.numpy())),
        num_heads = num_heads,
        activation = activation,
        residual = residual)
print(net)
path_grnn = save_path+'\\'+name+'_grnn'
paddle.save(net.state_dict(), path_grnn)


for i in tqdm(range(times)):
    val_loss_list = []
    val_acc_list = []
    test_acc_list = []  
    path_pre, val_loss, val_acc, test_acc = Trainer(net, path_grnn, mode='pre')
    
    for k in range(iteration_times): 
        val_loss_list.append(round(val_loss, 4))
        val_acc_list.append(round(val_acc, 4))
        test_acc_list.append(round(test_acc, 4))
        
        target = Upgrade(net, path_pre, isOnehot=True)
        path, val_loss, val_acc, test_acc = Trainer(net, path_grnn, mode='upgrade')
    val_loss_list.append(round(val_loss, 4))
    val_acc_list.append(round(val_acc, 4))
    test_acc_list.append(round(test_acc, 4))
    if i == 0:
        val_acc_sum_array = np.array(val_acc_list)
        test_acc_sum_array = np.array(test_acc_list)
    else:
        val_acc_sum_array = val_acc_sum_array + np.array(val_acc_list)
        test_acc_sum_array = test_acc_sum_array + np.array(test_acc_list)
    
    filename = save_path+'\\data\\'+name+'\\'+str(i)+'.txt'
    with open(filename, 'w') as f:
        f.write('val_loss_list:'+str(val_loss_list)+'\n')
        f.write('val_acc_list:'+str(val_acc_list)+'\n')
        f.write('test_acc_list:'+str(test_acc_list)+'\n')
    
    if i % 5 == 4:
        filename_all = save_path+'\\data\\'+name+'\\'+'summary'+'.txt'
        with open(filename_all, 'w') as f_all:
            f_all.write('name:'+name+'\n')
            f_all.write('hidden_size:'+str(hidden_size)+'\n')
            f_all.write('lr:'+str(lr)+'\n')
            f_all.write('activation:'+activation+'\n')
            f_all.write('l2_loss:'+str(l2_loss)+'\n')
            f_all.write('num_heads:'+str(num_heads)+'\n')
            f_all.write('feat_drop:'+str(feat_drop)+'\n')
            f_all.write('attn_drop:'+str(attn_drop)+'\n')
            f_all.write('input_drop:'+str(input_drop)+'\n')
            f_all.write('residual:'+str(residual)+'\n')
            f_all.write('Epoch:'+str(Epoch)+'\n')
            f_all.write('patience:'+str(patience)+'\n')
            f_all.write('val_acc_mean_list:'+str([round(temp, 4) for temp in list(val_acc_sum_array/(i+1))])+'\n')
            f_all.write('test_acc_mean_list:'+str([round(temp, 4) for temp in list(test_acc_sum_array/(i+1))])+'\n')
                
print('Finish')