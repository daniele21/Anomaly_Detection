# -*- coding: utf-8 -*-
#%%
from libraries.model.options import Options
from libraries.model.dataset import generateDataloader, getCifar10
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples
from libraries.model.adModel import AnomalyDetectionModel
from libraries.utils import Paths, getAnomIndexes, computeAnomError, computeNormError
paths = Paths()

from matplotlib import pyplot as plt
import numpy as np
import pickle

import torch
from torch.optim import Adam
from torchvision import transforms as Transforms
from torch.autograd import Variable
import sys
#%% LOAD OPTIONS
opt = Options()
opt.name = 'Ganom_1_v0.0'

opt.in_channels = 1 # GRAYSCALE
#opt.in_channels = 3 # RGB

opt.nFolders = 60
opt.patch_per_im = 500

#opt.nFolders = 30
#opt.patch_per_im = 1000

#opt.nFolders = 60
#opt.patch_per_im = 1500

#opt.n_extra_layers=2

opt.batch_size = 64
#opt.out_channels = 128

opt.descr = '-----'
opt.descr = 'augmentation - validation con norm - no weighting losses - thr over norm/anom'

#%% GENERATE DATASET

#cifar_dataloader = getCifar10(opt)
my_dataloader = generateDataloader(opt)

#%% LOAD DATASET

with open(paths.dataloaders + 'v1_60_500_30k.pickle', 'rb') as data:
    my_dataloader = pickle.load(data)

#%% SAVE DATASET
filename = 'v1_b&w_60-500-30k.pickle'
with open(paths.dataloaders + '/v1' + filename, 'wb') as f:
    pickle.dump(my_dataloader, f)
#%%

dataloader = my_dataloader
opt.dataset = 'steel dataset'
opt.augmentation = True
#dataloader = cifar_dataloader
#opt.dataset = 'cifar dataset'
#%%
trainloader = dataloader['train']
validLoader = dataloader['validation']
testloader = dataloader['test']
    
#%% MODEL
optimizer_gen = Adam
optimizer_discr = Adam

opt.lr_gen = 1*1e-04
opt.lr_discr = 1*1e-04
adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr,
                                trainloader, validLoader, testloader) 

#%% TUNING MODEL
tuning = adModel.tuneLearningRate(-6, -7, -6, -7)

#%% LOAD MODEL
optimizer_gen = Adam
optimizer_discr = Adam

adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr) 

opt.name = 'Ganom_v12.0'
nome_ckp = 'Ganom_v12.0_lr:1e-06|Epoch:201|Auc:0.901|Loss:178.0066.pth.tar'
path_file = paths.checkpoint_folder + opt.name + '/' + nome_ckp
print(path_file)
#adModel.loadCheckPoint(path_file)
adModel = torch.load(path_file)

#%% TRAINING MODEL
#opt.lr = 0.0001
#opt.lr_gen = 1e-05
#opt.lr_discr = 1e-05

#opt.epochs = 30
opt.patience = 5

opt.w_adv = 1
opt.w_con = 50
opt.w_enc = 1

#opt.w_adv = 1.306
#opt.w_con = 2.194
#opt.w_enc = 0.69

epochs = 30
adModel.train_model(epochs)

#%% RESUME LEARNING
##
##adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr) 
##adModel.loadTrainloader(trainloader)
##adModel.loadValidationLoader(validLoader)
#
#opt.name = 'Ganom_v6.0'
#nome_ckp = 'Ganom_v6.0_lr:1e-07|Epoch:87|Auc:0.879|Loss:228.2375.pth.tar'
#path_file = paths.checkpoint_folder + opt.name + '/' + nome_ckp
#print(path_file)
##adModel.loadCheckPoint(path_file)
#adModel = torch.load(path_file)
#%%
opt.lr_gen = 1*1e-04
#opt.lr_discr = 5*1e-05

adModel.opt.patience = 5
new_lr = 1e-04
adModel.model.optimizer_gen.param_groups[0]['lr'] = new_lr

adModel.resumeTraining(100)
#%% RESULTS
adModel.plotting()
adModel.evaluateRoc()

#%% TEST SET ERROR
anom_set = collectAnomalySamples(100)
norm_set = collectNormalSamples(90, 80)
#%% PREDICT

i = 0
#%%
image = validLoader.dataset.data[i]
target = validLoader.dataset.targets[i]
adModel.predict(image, target, i, verbose=1)
adModel.predict(image, target, 'nonono', verbose=1)
#print(score)
i += 1
#%% ANOMALY
anom_index = getAnomIndexes(validLoader.dataset.targets)
i = 0
#%%
image = validLoader.dataset.data[anom_index[i]]
target = validLoader.dataset.targets[anom_index[i]]
adModel.predict(image, target, i, verbose=1)
i += 1

#%% ANOM ERROR
anomalyError = computeAnomError(adModel, anom_set)
normalError = computeNormError(adModel, norm_set)

content = '- Anom_Error: {:.3f}'.format(anomalyError)
adModel.addInfo(content)

content = '\n- Norm_Error: {:.3f}'.format(normalError)
adModel.addInfo(content)

#%% PREDICTION
i = 150
#%%
image = validLoader.dataset.data[i]
image.shape
i += 1

image_tensor = torch.FloatTensor(image)
image_tensor.shape
image_tensor = Transforms.ToTensor()(image)
image_tensor.shape
image_tensor = image_tensor.unsqueeze_(0)
image_tensor.shape
image_var = Variable(image_tensor).cuda()
image_var.shape

#plt.imshow(image)

with torch.no_grad():
    x_prime, z, z_prime = adModel.model.forward_gen(image_var)

x_prime.shape
z.shape
z_prime.shape

output = x_prime.cpu().numpy()
output.shape

output[0].shape

final_output = np.transpose(output[0], (2,1,0))
final_output.shape

#plt.imshow(output)

#final_output = (output * 0.5) / 0.5
final_output = np.flip(final_output, 1)
final_output = np.rot90(final_output, 1)        

#        plt.imshow(final_output)

fig, [ax1, ax2] = plt.subplots(2,1, figsize=(15,15))
ax1.imshow(image)
ax2.imshow(final_output)

#%%
final_output = np.transpose(a, (2,1,0))
final_output.shape

c = b[0]

torch.mean(final_output)
torch.mean(b)
torch.mean(c)
#%%
image = trainloader.dataset.data[0]
image.shape
mean = np.mean(image)
mean
std = np.std(image)
std

r_ch = image[:,:,0]
g_ch = image[:,:,1]
b_ch = image[:,:,2]

r_mean = np.mean(r_ch)
g_mean = np.mean(g_ch)
b_mean = np.mean(b_ch)

r_mean
g_mean
b_mean

r_std = np.std(r_ch)
g_std = np.std(g_ch)
b_std = np.std(b_ch)

r_std
g_std
b_std

from PIL import Image

image = Image.fromarray(image)
grayImage = Transforms.Grayscale()(image)

image_tensor = Transforms.ToTensor()(image)
image_tensor.shape
torch.max(image_tensor)
torch.min(image_tensor)
