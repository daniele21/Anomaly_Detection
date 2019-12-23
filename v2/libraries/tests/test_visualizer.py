# -*- coding: utf-8 -*-

#%%
from model_options import Options
from visualizer import Visualizer
from visdom import Visdom
import numpy as np
from evaluate import evaluate
import torch
from math import pow
#%%
opt = Options()

#%%
visualizer = Visualizer(opt)

#%%
vis = Visdom(server=opt.display_server, port=opt.display_port)
#%%
train_index = np.arange(0,30, 0.5)
train_index.shape
train_index

valid_index = np.arange(0,30,0.5)
valid_index.shape
valid_index

x1 = np.linspace(0,30,60)
x1.shape
y1 = np.exp(x1)
y1.shape

x2 = np.linspace(0,30,60)
x2.shape
y2 = np.exp(x2) * 10
y2.shape

train_loss = y1
valid_loss = y2

visualizer.plot_loss(train_index, train_loss, valid_index, valid_loss, 'Loss')

#%%
x = np.random.randint(1,10,9)
x
np.mean(x.reshape(-1, 3), axis=1)

#%%
x = np.linspace(0,100,500)
x
y = x/2
y
vis.line(X=[1], Y=[[x, y]], win=1, name='1', update='append')
vis.line(X=x, Y=y, win=1, name = '2', update='append')

#%%
vis.line(np.random.randn(10), np.arange(0,10), win='line', name='3', update='append')
#vis.line()

#%% PLOTTING
loss1 = np.random.randint(0,10,20)
loss1.shape
loss1

loss2 = np.random.randint(0,10,20)
loss2.shape
loss2

loss = {}
loss['INDEX'] = np.arange(0,20)
loss['GENERATOR'] = loss1
loss['DISCRIMINATOR'] = loss2

visualizer.plot_loss(loss)

#%%
labels = torch.Tensor([0,0,0,1,1,1,1])
score = torch.Tensor([5.5, 5.7, 6, 2, 1, 7, 5])
ev = evaluate(labels, score)
ev
