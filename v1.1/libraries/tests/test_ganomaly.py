# -*- coding: utf-8 -*-
#%%
from libraries.model.evaluate import getThreshold, evaluateRoc
from libraries.model.options import Options
from libraries.model.dataset import generateDataloader, dataloaderSingleSet
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples
from libraries.model.adModel import AnomalyDetectionModel, LR_DECAY, LR_ONECYCLE, loadModel
from libraries.utils import Paths, getAnomIndexes, computeAnomError, computeNormError, ensure_folder
from libraries.model.postprocessing import distScores
from libraries.model import postprocessing as pp
from libraries.model.score import anomalyScoreFromDataset
paths = Paths()

from matplotlib import pyplot as plt
import numpy as np
import pickle

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

opt.batch_size = 64
#opt.out_channels = 128

opt.descr = '-----'
opt.descr = 'augmentation - validation con norm - no weighting losses - thr over norm/anom'

#%% GENERATE DATASET

opt.augmentation = True
my_dataloader = generateDataloader(opt)

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

#%% TUNING MODEL
tuning = adModel.tuneLearningRate(-6, -7, -6, -7)

#%% LOAD MODEL
#opt = Options()
opt.name = 'Ganom_v1_v3.0'
nome_ckp = 'CKP_Ganom_v1_v3.0_lr_5e-05_Epoch_36_Auc_0.783_Loss_155.4654.pth.tar'
path_file = paths.checkpoint_folder + opt.name + '/' + nome_ckp
print(path_file)

#adModel = torch.load(path_file)
adModel = loadModel(path_file, trainloader, validLoader, testloader)
#ckp = loadModel(, trainloader, validLoader, testloader)

#%% MULTI TASK LOSS WEIGHTS
adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr, optimizer_gen,
                                trainloader, validLoader, testloader) 

mtl = MultiLossWrapper(adModel, trainloader, 3)
optim = torch.optim.Adam(mtl.multiTaskLoss.parameters(), lr=1e-03)
mtl.train(40, optim)

#%% TRAINING MODEL
epochs = 30

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

#%% -------------POSTPROCESSING----------------

ckp = '/media/daniele/Data/Tesi/Thesis/Results/v1/Ganom_v1_v3_training_result/Ganom_v1_v3_best_ckp.pth.tar'
model = torch.load(ckp)

samples = [1000, 1002, 1004]
test_set = dataloaderSingleSet(samples, 1)

#masked_images = test_set.dataset.masked

save_folders = pp.setSaveFoldersResults(samples)
save_folders

as_map, gt_map, masked_map = anomalyScoreFromDataset(model, test_set, 8, 32)

# SAVING MASKED MAP
pp.saveMasks(masked_map, save_folders)
#%% CREATE FILTERED ANOMALY SCORES

kernel_params = {'conv':3,
                 'med': 3,
                 'gauss':3}

hist_params = {'bins':50,
               'range': (0,0.1),
               'figsize':(10,5)}

prob = 0.95

# FILTERED ANOMALY SCORES
as_maps = pp.computeFilters(as_map, kernel_params)
pp.saveFilters(as_maps, save_folders)

# THRESHOLD FOR EACH FILTER
thrs = pp.computeThresholds(as_map, kernel_params, hist_params, prob, save_folders['general'])

#%% EVALUATION ROC CURVE

auc_std, best_thr_std = evaluateRoc(std_as.ravel(), gt_map.ravel(), info='Standard', thr=std_thr)
auc_conv, best_thr_conv = evaluateRoc(conv_as.ravel(), gt_map.ravel(), info='Conv', thr=conv_thr)
auc_med, best_thr_med = evaluateRoc(med_as.ravel(), gt_map.ravel(), info='Median', thr=med_thr)
auc_gauss, best_thr_gauss = evaluateRoc(gauss_as.ravel(), gt_map.ravel(), info='Gaussian', thr=gauss_thr)

#%% TUNING KERNELS

pp.tuning_conv_filter(as_map, gt_map)

pp.tuning_med_filter(as_map, gt_map)

pp.tuning_gauss_filter(as_map, gt_map)

#%%  STEPS FOR EVALUATING

index = 1
keys = list(save_folders.keys())
key = keys[index]
key

# ANOMALY DETECTION MAPS
anomaly_maps, res = pp.compute_anomalies_all_filters(index, gt_map, as_maps,
                                                    thrs, save_folders[key])

pp.saveAnomalyMaps(anomaly_maps, save_folders[key])

# OVERLAPPING GT
full_masked_maps = pp.overlapAnomalies(index, masked_map, anomaly_maps)
pp.saveFullMasked(full_masked_maps, save_folders[key])
#%% EVALUATION
evaluation = pp.resultsPerEvaluation(res)
evaluation

pp.writeResults(res, bests, key, save_folders[key])


bests = pp.best_performance(evaluation)
bests
#%% COMPLETE EVALUATION EXAMPLE
for i in range(len(samples)):
    key = str(samples[i])
    
    anomaly_maps, full_masked_maps, ev, bests = pp.complete_evaluation(i, gt_map, as_maps,
                                                                       masked_map, thrs,
                                                                       save_folders[key])


index = 0
anomaly_map, masked, ev, bests = pp.complete_evaluation(index, gt_map,as_filters, thr_filters,
                                                masked_images)
pp.plotAnomalies(as_filters, anomaly_map, masked, index, bests=bests)

index = 1
anomaly_map, masked, ev, bests = pp.complete_evaluation(index, gt_map, as_filters, thr_filters,
                                                masked_images)
pp.plotAnomalies(as_filters, anomaly_map, masked, index, bests=bests)

index = 2
anomaly_map, masked, ev, bests = pp.complete_evaluation(index, gt_map, as_filters, thr_filters,
                                                masked_images)
pp.plotAnomalies(as_filters, anomaly_map, masked, index, bests=bests)

index = 3
anomaly_map, masked, ev, bests = pp.complete_evaluation(index, gt_map, as_filters, thr_filters,
                                                masked_images)
pp.plotAnomalies(as_filters, anomaly_map, masked, index, bests=bests)

#%%
filters = ['standard', 'conv', 'med', 'gauss'] 
#%%
for f in filters:
    plt.imshow(anomaly_map[f]['1002'])
    plt.show()
#%%
for f in filters:
    plt.imshow(final_masks[f])
    plt.show()
#%%
for i in range(len(masked_images)):
    plt.imshow(masked_images[i])
    plt.show()
#%%
for f in filters:
    plt.imshow(final_masks[f])
    plt.show()
#%%
for key in full_masked_map:
    plt.imshow(full_masked_map[key])
    plt.show()
    
#%%
for key in as_map:
    plt.imshow(as_map[key])
    plt.show()
#%%
a = masked['conv']
#plt.title('Conv')
#fig = plt.figure(frameon = False)
#fig.set_size_inches(7, 2)
fig = plt.figure(figsize=(18, 3))
plt.axis('off')
plt.imshow(a)
plt.savefig('./prova.png', 
               transparent=True,
               dpi=1000)
plt.close(fig)
plt.show()

    
    
    
    