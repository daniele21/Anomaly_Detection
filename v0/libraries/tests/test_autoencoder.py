

#%% IMPORTS
import torch
from torch.autograd import Variable
from torch.optim import Adam

import numpy as np
from matplotlib import pyplot as plt

from libraries.model.options import Options
from libraries.model.autoencoder import AutoencoderModel
from libraries.model.network import weights_init
from libraries.model.dataset import generateDataloader, collectAnomalySamples, collectNormalSamples
from libraries.utils import Paths, Checkpoint, getAnomIndexes, computeAnomError, computeNormError

paths = Paths()
#%%

opt = Options()
opt.name = 'AE_v2.2'
opt.anom_perc = 0.4
opt.in_channels = 1
opt.nFolders = 200
opt.patch_per_im = 500

opt.descr = '-----'
opt.descr = 'xavier init'
#%% DATASET

#cifar_dataloader = getCifar10(opt)
my_dataloader = generateDataloader(opt)

#%%
dataloader = my_dataloader
opt.dataset = 'steel dataset'

#dataloader = cifar_dataloader
#opt.dataset = 'cifar dataset'
#%%
trainloader = dataloader['train']

try:
    validLoader = dataloader['test']
except:    
    validLoader = dataloader['validation']
    
#%%
optimizer = Adam
aeModel = AutoencoderModel(opt, optimizer, trainloader, validLoader)
aeModel.model.apply(weights_init)


#%% LOAD MODEL
opt = Options()
opt.name = 'AE_v3.1'
nome_ckp = 'AE_v3.1_lr:1e-05|Epoch:29|Loss:0.0448.pth.tar'
path_file = paths.checkpoint_folder + opt.name + '/' +nome_ckp
aeModel = torch.load(path_file)

#aeModel.loadModel(ckp)


#%% TRAINING MODEL
opt.epochs = 30
opt.patience = 5
#opt.lr = 6.3*1e-5
opt.lr = 0.00001
aeModel.train_autoencoder(opt)

#%% PREDICTION
anom_set = collectAnomalySamples(100)
norm_set = collectNormalSamples(100,80)
# NORMAL SAMPLES
i=0
#%% JUST VISUALIZING
image = trainloader.dataset.data[i]
image.shape
score = aeModel.predict(image)
i += 10

#%% SAVING
info = '(from Normal)' + str(i)
image = trainloader.dataset.data[i]
image.shape
score = aeModel.predict(image, info)
i += 10
#%%
index = getAnomIndexes(validLoader.dataset.targets)

# ANOMALY SAMPLES
i = 0
#%% VALIDATION SET
info = '(from Anomaly)' + str(i)
image = validLoader.dataset.data[index[i]]
aeModel.predict(image, info)
i += 8

#%% ANOMALY SET
i = 0
#%%
image = anom_set[i]
aeModel.predict(image, info=i)
i += 5
#%%
anomalyError = computeAnomError(aeModel, anom_set)
normalError = computeNormError(aeModel, norm_set)

content = '- Anom_Error: {:.3f}\n'.format(anomalyError)
aeModel.addInfo(content)
content = '- Norm_Error: {:.3f}'.format(normalError)
aeModel.addInfo(content)
#%% PLOTTING
#aeModel.avg_loss['validation'][-1]
aeModel.plotting()
aeModel.evaluateRoc()

#%% TUNING

aeModel.tuneLearningRate(inf_bound=-7, sup_bound=-5)

#%% SAVING CHECKPOINT

a = Checkpoint('model', 'optim', 12.23, 5, 'file.htiwod', '---')
a.save()

#%% LOADING EXISTENT MODEL

path_file = paths.checkpoint_folder + 'My Ganomaly_ckp_19_6.849.pth.tar'
path_file
a = torch.load(path_file)
a

model = a.model
optimizer = a.optimizer
type(optimizer)

#%% PREDICTION
image = validLoader.dataset.data[10]
image.shape
label = validLoader.dataset.targets[10]
label.shape
label

image_tensor = torch.FloatTensor(image)

image_tensor.shape
image_tensor = image_tensor.unsqueeze_(0)
image_tensor.shape
image_tensor

image_var = Variable(image_tensor).cuda()
image_var.shape

plt.imshow(image)
#%%
with torch.no_grad():
        pred = aeModel.model(image_var)

pred.shape

output = pred.cpu().numpy()
output[0].shape
output = np.transpose(output[0], (2,1,0))
#output = output.view(0, 2,1,1)
output.shape

#output = 0.5 * (output + 1)
final_output = (output * 0.5) / 0.5
final_output = np.flip(final_output, 1)
final_output = np.rot90(final_output, 1)

output.shape
#plt.imshow(final_output)

fig, [ax1, ax2] = plt.subplots(2,1, figsize=(5,5))
#plt.figure()
ax1.imshow(image)
ax2.imshow(final_output)
plt.show()

#%%
for data in trainloader:
    
    break

#%%

# IMAGES
data[0]
data[0].shape

# LABELS
data[1]
data[1].shape

tensor_image = torch.Tensor(data[0]).cuda()
tensor_image.shape
tensor_image
