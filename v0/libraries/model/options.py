# -*- coding: utf-8 -*-
#%%
#from dataset import loadData
from libraries.model.dataset import newLoadData
#%%
class Options():

    def __init__(self,
                 # DATASET
#                 path_normal = '/media/daniele/Data/Tesi/Practice/Dataset/my_dataset/patches/Normal/',
#                 path_anom   = '/media/daniele/Data/Tesi/Practice/Dataset/my_dataset/patches/Anomalous/',
                 anom_perc   = 0.4,
                 nFolders    = 5,
                 startFolder = 1,
                 endFolder   = 1000,
                 patch_per_im= 2000,
                 transforms  = None,
                 batch_size  = 64,
                 split       = 0.8,
                 n_workers   = 4,
                 
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
                 w_adv          = 1,
                 w_con          = 50,
                 w_enc          = 1,
                 kernel_size    = 3,
                 
                 dataset        = '',
                 descr          = '',
                 
                 # VISUALIZATION
                 loss_per_epoch = 20,
                 printing_freq  = 5,
                 avg_freq       = 5,
                 display        = True, 
                 save_image_freq= 100, 
		         save_test_images = False,     
                 display_server = 'http://localhost', 
                 display_port   = 8097
                 ):
        
        
        #NETWORK
        self.img_size       = img_size
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.z_size         = z_size
        self.n_extra_layers = n_extra_layers
        
        # DATASET
#        self.path_normal    = path_normal
#        self.path_anom      = path_anom
        self.anom_perc      = anom_perc   
        self.nFolders       = nFolders
        self.startFolder    = startFolder
        self.endFolder      = endFolder
        self.patch_per_im   = patch_per_im
        self.transforms     = transforms
        self.batch_size     = batch_size
        self.split          = split
        self.n_workers      = n_workers
        
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
        self.w_adv          = w_adv
        self.w_con          = w_con
        self.w_enc          = w_enc
        self.kernel_size    = kernel_size
        
        self.dataset        = dataset
        self.descr          = descr
        
        # VISUALIZATION
        self.loss_per_epoch = loss_per_epoch
        self.printing_freq  = printing_freq
        self.avg_freq       = avg_freq
        self.display        = display
        self.save_image_freq = save_image_freq
        self.save_test_images = save_test_images
        self.display_server = display_server
        self.display_port   = display_port
        
        self.isTrain = True


    def loadDatasets(self):
        train, train_targets, val, val_targets = newLoadData(self)
#        train, train_targets, val, val_targets = loadData(self)
        
        self.train_data = train
        self.train_targets = train_targets
        self.validation_data = val
        self.validation_targets = val_targets
        
        self.loadedData = True
        
        
        
        
        
        
        
        
        
        
