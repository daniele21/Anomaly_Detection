# -*- coding: utf-8 -*-
#%%
from libraries.MultiTaskLoss import MultiLossWrapper
from libraries.model.options import Options
from libraries.model.dataset import generateDataloaderTL, generateDataloaderPerDefect, generateDataloaderAS
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples
from libraries.model.adModel import AnomalyDetectionModel, loadModel
from libraries.utils import Paths, getAnomIndexes, computeAnomError, computeNormError
from libraries.model.filterModel import FilterModel
from libraries.model import postprocessing as pp
from libraries.model import score
paths = Paths()

from matplotlib import pyplot as plt
import numpy as np
import pickle
import cv2

import torch
from torch.optim import Adam
from torchvision import transforms as Transforms
from torch.autograd import Variable

#%% LOAD OPTIONS
opt = Options()
opt.name = 'Ganom_v1_v4.0'

#opt.in_channels = 1 # GRAYSCALE
opt.in_channels = 3 # RGB

opt.nFolders = 2
opt.patch_per_im = 500

#opt.nFolders = 30
#opt.patch_per_im = 1000

#opt.nFolders = 60
#opt.patch_per_im = 1500

#opt.n_extra_layers=2
opt.tl = 'resnet18'
opt.batch_size = 64
#opt.out_channels = 128

opt.descr = '-----'
opt.descr = 'augmentation - validation con norm - no weighting losses - thr over norm/anom'

#%% GENERATE DATASET
opt.TL_size = 128
opt.augmentation = True
my_dataloader = generateDataloaderTL(opt)

dataloader = my_dataloader
opt.dataset = 'steel dataset'
trainloader = dataloader['train']
validLoader = dataloader['validation']
testloader = dataloader['test']
#%%

for image, image_TL, target in validLoader:
#    print(image)
#    print(image_TL)
#    print(target)
    break

image = image
#image = image_TL

final_output = np.transpose(image[0], (2,1,0))

final_output = (final_output* 0.5) + 0.5
#final_output = np.flip(final_output, 1)
#final_output = np.rot90(final_output, 1)        

plt.imshow(final_output)        
#%% LOAD DATASET

with open(paths.dataloaders + 'v1_aug_60-500-30k.pickle', 'rb') as data:
    my_dataloader = pickle.load(data)

#%% SAVE DATASET
filename = 'v1_60-500-30k.pickle'
with open(paths.dataloaders + '/v1' + filename, 'wb') as f:
    pickle.dump(my_dataloader, f)
#%%

dataloader = my_dataloader
opt.dataset = 'steel dataset'

#%%
trainloader = dataloader['train']
validLoader = dataloader['validation']
testloader = dataloader['test']
    
#%% MODEL OPTIONS
optimizer_gen = Adam
optimizer_discr = Adam

opt.lr_gen = 5*1e-05
opt.lr_discr = 5*1e-05

opt.patience = 5

opt.w_adv = 1
opt.w_con = 50
opt.w_enc = 1

opt.weightedLosses = False

epochs = 30

epochs = 5

# NORM GRAD
#opt.weightedLosses=False
#optimizer_weights = None

opt.weightedLosses=False
optimizer_weights = optimizer_gen

# WEIGHTED LOSS TRAINING
#opt.multiTaskLoss = True
opt.multiTaskLoss = False

adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr, optimizer_weights,
                                trainloader, validLoader, testloader) 

#adModel.setLRscheduler(LR_ONECYCLE, 1e-04, epochs)
#adModel.setLRscheduler('decay', 0.3, epochs)

adModel.train_model(epochs)


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
norm_set = collectNormalSamples(100, 200)

filename = 'v1_anom-set_22k.pickle'
with open(paths.dataloaders + '/v1' + filename, 'wb') as f:
    pickle.dump(anom_set, f)
    
filename = 'v1_norm-set_20k.pickle'
with open(paths.dataloaders + '/v1' + filename, 'wb') as f:
    pickle.dump(norm_set, f)
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

#%% POSTPROCESSING

ckp = 'MODEL_Ganom_v2_v2_vgg_2enc_no-frezeeing_lr_7.000000000000001e-05_Epoch_250_Loss_187.2821.pth.tar'
path = '/media/daniele/Data/Tesi/Thesis/Results/v2/'

model = torch.load(path + ckp) 

#%%
thr = 0.046
model.threshold = thr

#%% FILTER MODEL

opt = Options(in_channels=1, out_channels=1, batch_size=1)

samples = [1000, 1001]
defect = 3
filter_data = generateDataloaderPerDefect(opt, model,
                                          samples, stride=8)
#%%
defect = 3

optim = torch.optim.Adam
trainloader = filter_data
validloader = filter_data

#%%

k = 5
opt.lr = 1e-02
filter_model = FilterModel(optim, trainloader, validloader, opt, k)

filter_model.train_model(100, model.threshold)


#%%

kernel = filter_model.model.conv.weight
kernel = kernel.cpu().detach().numpy()
kernel = kernel.reshape(k,k)

plt.imshow(kernel)
kernel

#%% COMPARISON FILTERS
image = trainloader.dataset.data[0]
plt.imshow(image)
plt.show()

label = trainloader.dataset.targets[0]
plt.imshow(label)
plt.show()

#%%
images = pp.loadDefectImages()
samples = images['n_images']
defects = images['Defect']
len(samples)

save_folders = pp.setSaveFoldersResults(samples, folder_extra_name='prova')
save_folders

opt = Options(in_channels=3, batch_size=1)
test_set = generateDataloaderAS(opt, model, samples[0:2])
#%%
as_images = test_set.dataset.score
masks = test_set.dataset.targets
images = test_set.dataset.data
samples = test_set.dataset.samples

for i in range(len(test_set)):
    as_image = as_images[i]
    
    conv_filter_model = pp.convFilterScores(image, kernel)
    conv_filter = pp.convFilterScores(image, pp.createKernel(3,2))

#%%
hist_params = {'bins':50,
               'range': (0,0.1),
               'figsize':(10,5)}

prob = 0.95

thr = pp.computeThreshold(conv_filter_model, kernel, hist_params, prob)
#pp.computeThreshold(conv_filter_model, kernel, hist_params, prob, save_folders['general'])

#%%
images = pp.loadDefectImages()
samples = images['n_images']
defects = images['Defect']
len(samples)

save_folders = pp.setSaveFoldersResults(samples, folder_extra_name='prova')
save_folders

opt = Options(in_channels=3, batch_size=1)
test_set = generateDataloaderAS(opt, model, samples[0:1])
#%%
as_images = test_set.dataset.score
masks = test_set.dataset.targets
images = test_set.dataset.data
samples = test_set.dataset.samples
#%%
for i in range(len(test_set)):
    
    image = images[i]
    as_image = as_images[i]
    mask = masks[i]
    sample = samples[i]
    
    plt.imshow(as_image)
    plt.show()
    plt.imshow(mask)
    plt.show()
    
    pp.saveMask(mask, sample, save_folders)
    
    # FILTERING ANOMALY SCORES
#    conv_filter_model = pp.convFilterScores(as_image, kernel)
#    pp.saveFilter(sample, conv_filter, save_folders, filter_name='Learned_Conv')
    
    conv_filter = pp.convFilterScores(as_image, pp.createKernel(3,2))    
    pp.saveFilter(sample, conv_filter, save_folders, filter_name='Costant-Conv')
    
    # ANOMALY DETECTION AND RESULTS
#    anom_im, res = pp.compute_filter_anomalies(conv_filter_model, mask, model.threshold)
    anom_im, res = pp.compute_filter_anomalies(conv_filter, mask, model.threshold)   

    pp.saveAnomalyMap(anom_im, sample, save_folders, filter_name='Constant-conv')

    plt.imshow(anom_im)
    plt.show()
    
    plt.imshow(mask)
    plt.show()
    
    full_mask = pp.overlapAnomalies_single(anom_im, mask, image)

    pp.saveFullMask(full_mask, sample, save_folders, filter_name='Constant-conv')

    plt.imshow(full_mask)
    plt.show()
    
    evaluation = pp.resultsPerEvaluation_single(res)
    evaluation    
    
#    pp.res_table = pp.res_table_init(save_folders['general'])
    
    pp.fillResultTable_single(res, sample, save_folders)
    
