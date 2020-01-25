# -*- coding: utf-8 -*-
#%%
#from dataset import loadData
from libraries.model.dataset import loadDataset
#%%
class Options():

    def __init__(self,
                 # DATASET
                 nFolders    = 5,
                 startFolder = 1,
                 endFolder   = 2000,
                 patch_per_im= 2000,
                 transforms  = None,
                 batch_size  = 64,
                 split       = 0.8,
                 n_workers   = 8,
                 augmentation= True,
                 shape = 32,
                 
                 # NETWORK
                 img_size       = 32,
                 in_channels    = 3,   # 1=GRAYSCALE   3=RGB 
                 out_channels   = 64, 
                 z_size         = 100, 
                 n_extra_layers = 0,
                 
                 # MODEL
                 name           = 'My Ganomaly',
                 seed           = -1, 
                 epochs         = 10, 
                 patience       = 3,
                 beta1          = 0.5, 
                 lr             = 0.0005,
                 lr_gen         = 0.0002, 
                 lr_discr       = 0.0001,
                 output_dir     = '/media/daniele/Data/Tesi/Practice/Code/ganomaly/ganomaly-master/output', 
                 load_weights   = True, 
                 phase          = 'train', 
                 resume         = '',
                 alpha          = 0.15,
                 weightedLosses = False,
                 w_adv          = 1,
                 w_con          = 50,
                 w_enc          = 1,
                 multiTaskLoss  = False,
                 kernel_size    = 3,
                 sigma          = 1,
                 tl             = 'vgg16',
                 
                 dataset        = '',
                 descr          = '',

                 ):
        
        
        #NETWORK
        self.img_size       = img_size
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.z_size         = z_size
        self.n_extra_layers = n_extra_layers
        
        # DATASET  
        self.nFolders       = nFolders
        self.startFolder    = startFolder
        self.endFolder      = endFolder
        self.patch_per_im   = patch_per_im
        self.transforms     = transforms
        self.batch_size     = batch_size
        self.split          = split
        self.n_workers      = n_workers
        self.augmentation   = augmentation
        self.shape = shape
        
        self.train_data = []
        self.train_targets = []
        self.validation_data = []
        self.validation_targets = []
        
        self.loadedData = False
        
        # MODEL
        self.seed           = seed
        self.name           = name
        self.patience       = patience
        self.epochs         = epochs
        self.lr             = lr
        self.lr_gen        = lr_gen
        self.lr_discr       = lr_discr
        self.beta1          = beta1
        self.load_weights   = load_weights
        self.phase          = phase
        self.output_dir     = output_dir
        self.resume         = resume
        self.alpha          = alpha
        self.weightedLosses = weightedLosses
        self.w_adv          = w_adv
        self.w_con          = w_con
        self.w_enc          = w_enc
        self.multiTaskLoss = multiTaskLoss
        self.kernel_size    = kernel_size
        self.sigma          = sigma
        self.tl             = tl
        
        self.dataset        = dataset
        self.descr          = descr
        
        self.isTrain = True


    def loadDatasets(self):
        train, validation, test = loadDataset(self, test='mixed')
#        train, validation, test = loadDataset(self, test='normal')
        
#        train, train_targets, val, val_targets, test, test_targets = loadDataNormAnonm(self)
#        train, train_targets, val, val_targets, test, test_targets = loadDatasetAllNormals(self)
#        train, train_targets, val, val_targets = loadData(self)
        
#        self.train_data = train
#        self.train_targets = train_targets
#        self.validation_data = val
#        self.validation_targets = val_targets
#        self.test_data = test
#        self.test_targets = test_targets
        
        self.training_set = train
        self.validation_set = validation
        self.test_set = test
        
        self.loadedData = True
        
#%%
class FullImagesOptions():
    
    def __init__(self,
                 # DATASET
                 augmentation   = True,
                 batch_size     = 16,
                 split          = 0.7,
                 n_workers      = 8,
                 start = 0,
                 end = 100,
                 shape = 64,
                 
                 name           = 'My_Ganomaly',
                 in_channels = 3,
                 ):
        
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.split = split
        self.n_workers = n_workers
        self.start = start
        self.end = end
        self.shape = shape
        
        self.name = name
        self.in_channels = in_channels

        
        
        
        
        
        
        
        
