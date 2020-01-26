# -*- coding: utf-8 -*-
#%% IMPORTS
import time
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

import torch.utils.data
from torchvision import transforms as Transforms
from torch.autograd import Variable

from libraries.MultiTaskLoss import MultiLossWrapper
from libraries.model.ganomaly_network import GanomalyModel
from libraries.model.evaluate import evaluate
from libraries.utils import EarlyStopping, saveInfoGanomaly, addInfoGanomaly, LR_decay
from libraries.utils import Paths, ensure_folder, getNmeans, Checkpoint
from libraries.model.postprocessing import convFilterScores, medFilterScores, gaussFilterScores
paths = Paths()

from libraries.dataset_package.dataset_manager import generatePatches
#%% CONSTANTS
GENERATOR = 'GENERATOR'
DISCRIMINATOR = 'DISCRIMINATOR'

LR_DECAY = 'decay'
LR_ONECYCLE = 'oneCycle'

device = torch.device('cuda:0')
#%%

#def loadModel(filename):
#        
#    model_name = filename.split('_')[0] + '_' + filename.split('_')[1]
#    path_file = paths.checkpoint_folder + model_name + '/' + filename
#    
#    return torch.load(path_file)

def loadModel(filename, trainloader, validloader, testloader):
        
#    model_name = filename.split('_')[1] + '_' + filename.split('_')[3]
#    path_file = paths.checkpoint_folder + model_name + '/' + filename
       
    path_file = filename
    ckp = torch.load(path_file)
    
    adModel = AnomalyDetectionModel(ckp.opt)
    adModel.initLoaders(trainloader, validloader, testloader)
    adModel.folder_save = ckp.folder_save
#    adModel.train_loss = ckp.trainloss
#    adModel.val_loss = ckp.validloss
    adModel.epoch = ckp.epoch
    adModel.auc = ckp.auc
    adModel.threshold = ckp.threshold
    adModel.gt_labels = ckp.gt_labels
    adModel.scores = ckp.scores
    
    return adModel

class AnomalyDetectionModel():
    
    def __init__(self, opt, optim_gen=None, optim_discr=None, optim_weights=None,
                 trainloader=None, validationloader=None, testloader=None):
        
        self.model              = GanomalyModel(opt)
#        optimizer_gen           = optim_gen(self.model.generator.parameters(), opt.lr_gen)
#        optimizer_discr         = optim_discr(self.model.discriminator.parameters(), opt.lr_discr)
#        optimizer_weights       = optim_gen(self.model.w_losses, opt.lr_gen)
        self.model.init_optim(optim_gen, optim_discr, optim_weights)
        self.initLoaders(trainloader, validationloader, testloader)
        self.opt                = opt
        self.mtl = MultiLossWrapper(self.model, trainloader, 3)
        
    
    def initLoaders(self, trainloader, validloader, testloader):
        self.trainloader        = trainloader
        self.validationloader   = validloader
        self.testloader         = testloader
    
    def get_loss(self):
        
        losses = OrderedDict([
                    
                    ('loss_gen', self.model.err_gen.item()),
                    ('loss_discr', self.model.err_discr.item()),
                    ('loss_gen_adv', self.model.err_gen_adv.item()),
                    ('loss_gen_con', self.model.err_gen_con.item()),
                    ('loss_gen_enc', self.model.err_gen_enc.item()),
                
                ])
    
        return losses

    def _trainOneEpoch(self):    

        self.model.train()
        train_loss = {}
        train_loss[GENERATOR] = []
        train_loss[DISCRIMINATOR] = []
        
        adv_loss = []
        con_loss = []
        enc_loss = []        
        
        n_iter = len(self.trainloader)
        
        start = time.time()
        
#        for images, labels in tqdm(self.trainloader, leave=True, total=n_iter, desc='Training', file = sys.stdout):
        
#        for param in self.model.generator.parameters():
#            param.requires_grad=False
        
        if(self.opt.multiTaskLoss):
            w_adv, w_con, w_enc = self.mtl.train(20, patience=1)
            self.model.setWeights(w_adv, w_con, w_enc)
            print(self.model.w_losses)
        
#        for param in self.model.generator.parameters():
#            param.requires_grad=True
        
        for images, images_TL, labels in self.trainloader:

            x = torch.Tensor(images).cuda()
            x_TL = torch.Tensor(images_TL).cuda()
                
            # GENERATOR FORWARD
#            x_prime, z, z_prime = self.model.forward_gen(x)
            x_prime, z, z_prime = self.model.forward_gen(x_TL)

            # DISCRIMINATOR FORWARD
            pred_real, feat_real, pred_fake, feat_fake = self.model.forward_discr(x, x_prime) 
            
            # GENERATOR LOSS
            loss_gen, losses = self.model.loss_function_gen(x, x_prime, z, z_prime, feat_fake, feat_real, self.opt)
#            print(loss_gen.item())
            if(self.epoch == 0):
                self.l0 = losses
            
            
            # DISCRIMINATOR LOSS
            loss_discr = self.model.loss_function_discr(pred_real, pred_fake)
            
            # BACKWARDS
            self.model.optimize_gen(loss_gen, self.l0)
            self.model.optimize_discr(loss_discr)
            
            train_loss[GENERATOR].append(loss_gen.item()*images.size(0))
#            print(train_loss[GENERATOR])
            train_loss[DISCRIMINATOR].append(loss_discr.item()*images.size(0))
            adv_loss.append(losses[0].item()*images.size(0))
            con_loss.append(losses[1].item()*images.size(0))
            enc_loss.append(losses[2].item()*images.size(0))
            
            # UPDATE LR SCHEDULER
            if(self.lr_policy == LR_ONECYCLE):
                self.lr_scheduler_gen.step()
                self.lr_scheduler_discr.step()
        
        spent_time = time.time() - start
        
        if(self.opt.weightedLosses):
            print('\n------------------------\n')
            print('> Loss weights')
            try:
                print('w_adv: {}'.format(self.model.w_losses[0]))
                print('w_con: {}'.format(self.model.w_losses[0]))
                print('w_enc: {}'.format(self.model.w_losses[0]))
            except:
                print('w_adv: {}'.format(self.model.w_losses[0]))
                print('w_con: {}'.format(self.model.w_losses[0]))
                print('w_enc: {}'.format(self.model.w_losses[0]))
            print('----------------------------')
            
        return train_loss, [adv_loss, con_loss, enc_loss], spent_time
            
    def _validation(self):
        print('wdinfuqieruòvbqiòbvqiuvbqiuebvqiuebviuqrbv')
        curr_epoch = 0
        steps = 0
        
        valid_loss = {}
        valid_loss[GENERATOR] = []
        valid_loss[DISCRIMINATOR] = []
        
        adv_loss = []
        con_loss = []
        enc_loss = []
        
        n_iter = len(self.validationloader)
        
        start = time.time()
        
        self.model.evaluate()
        with torch.no_grad():
            
#            for images, labels in tqdm(self.validationloader, leave=True, total=n_iter, desc='Validation', file = sys.stdout):
            for images, images_TL, labels in self.validationloader:
                
                steps += 1
                curr_epoch += self.opt.batch_size
                
#                x = torch.Tensor(images).cuda()
                x = Variable(images).cuda()
                x_TL = torch.Tensor(images_TL).cuda()
                
                # GENERATOR FORWARD
                x_prime, z, z_prime = self.model.forward_gen(x_TL)
                # DISCRIMINATOR FORWARD
                pred_real, feat_real, pred_fake, feat_fake = self.model.forward_discr(x, x_prime)    
                
                # GENERATOR LOSS
                loss_gen, losses = self.model.loss_function_gen(x, x_prime, z, z_prime, feat_fake, feat_real, self.opt)
                print(loss_gen)
                print(loss_gen.item())
                print('---------------------------')
                # DISCRIMINATOR LOSS
                loss_discr = self.model.loss_function_discr(pred_real, pred_fake)
#                print('-----------aaa {}'.format(loss_gen.item()))
                valid_loss[GENERATOR].append(loss_gen.item()*images.size(0))
                valid_loss[DISCRIMINATOR].append(loss_discr.item()*images.size(0))

                adv_loss.append(losses[0].item()*images.size(0))
                con_loss.append(losses[1].item()*images.size(0))
                enc_loss.append(losses[2].item()*images.size(0))
        
            spent_time = time.time() - start
            
            # UPDATE LR SCHEDULER
            if(self.lr_policy == LR_DECAY):
                self.lr_scheduler_gen.step()
                self.lr_scheduler_discr.step()
                
        return valid_loss, [adv_loss, con_loss, enc_loss], spent_time
        
    def _test(self, normal_score=False):
        
        start = time.time()
        
        test_loader = self.testloader

        valid_loader = self.validationloader
        
        with torch.no_grad():
            
            
            curr_epoch = 0
            times = []
#            n_iter = len(self.validationloader)

            anomaly_scores = torch.zeros(size=(len(test_loader.dataset),), dtype=torch.float32, device=device)
            normal_scores = torch.zeros(size=(len(valid_loader.dataset),), dtype=torch.float32, device=device)
            
            gt_labels = torch.zeros(size=(len(test_loader.dataset),), dtype=torch.long,    device=device)
            normal_gt_labels = torch.zeros(size=(len(valid_loader.dataset),), dtype=torch.long,    device=device)
            
            
            # ANOMALY SCORE COMPUTED ON TEST SET(NORMAL + ANORMAL SAMPLES)
            i=0
            for images, labels in test_loader:
                
                curr_epoch += self.opt.batch_size
                
                time_in = time.time()
                
                x = torch.Tensor(images).cuda()
                tensor_labels = torch.Tensor(labels).cuda()                
                
                
                _, z, z_prime = self.model.forward_gen(x)
#                print('\n----Z----')
#                                       torch.Size([64, 100, 1, 1])
#                print(z.shape)
                
                # ANOMALY SCORE
                score = torch.mean(torch.pow((z-z_prime), 2), dim=1)
                
#                print('Score: ', score)
                
                time_out = time.time()
                                
                
                anomaly_scores[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
                gt_labels[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = tensor_labels.reshape(score.size(0))

                times.append(time_out - time_in)

                i += 1
            
            # NORMAL SCORE COMPUTED ON VALIDATION SET(ALL NORMAL SAMPLES)
            i=0
            if(normal_score):
                for images, labels in valid_loader:
                    
                    curr_epoch += self.opt.batch_size
                    
                    time_in = time.time()
                    
                    x = torch.Tensor(images).cuda()
                    tensor_labels = torch.Tensor(labels).cuda()                
                                    
                    _, z, z_prime = self.model.forward_gen(x)
                    
                    # NORMAL SCORE
                    score = torch.mean(torch.pow((z-z_prime), 2), dim=1)
    
                    time_out = time.time()
                                    
                    
                    normal_scores[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
                    normal_gt_labels[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = tensor_labels.reshape(score.size(0))
    
                    times.append(time_out - time_in)
    
                    i += 1 
            #-----------------------------------------------------------------------------------------------
            
            # NORMALIZATION - Scale error vector between [0, 1]
            anomaly_scores_norm = (anomaly_scores - torch.min(anomaly_scores)) / (torch.max(anomaly_scores) - torch.min(anomaly_scores))
            
            auc_norm, threshold_norm = evaluate(gt_labels, anomaly_scores_norm, info='2_norm',
                                                folder_save=self.results_folder, plot=True)
            
            # WITHOUT NORMALIZATION
            auc, threshold = evaluate(gt_labels, anomaly_scores,
                                      folder_save=self.results_folder, plot=True, info='1_standard')

            # CONV ANOMALY SCORE
            kernel_size = self.opt.kernel_size
            conv_anom_scores = convFilterScores(anomaly_scores, kernel_size)            
            auc_conv, conv_threshold = evaluate(gt_labels, conv_anom_scores, plot=True,
                                                folder_save=self.results_folder, info='4_conv')

            # MEDIAN ANOMALY SCORE
            kernel_size = self.opt.kernel_size
            median_anom_scores = medFilterScores(anomaly_scores, kernel_size)
            auc_median, median_threshold = evaluate(gt_labels, median_anom_scores, info='3_median',
                                                    plot=True, folder_save=self.results_folder)
            
            # GAUSSIAN ANOMALY SCORE
            sigma = self.opt.sigma
            gauss_anom_scores = gaussFilterScores(anomaly_scores, sigma)
            auc_gauss, gauss_threshold = evaluate(gt_labels, gauss_anom_scores, info='5_gauss',
                                                  plot=True, folder_save=self.results_folder)

            
            performance_norm = dict({'AUC':auc_norm,
                                'Threshold':threshold_norm})
    
            performance_stand = dict({'AUC':auc,
                                'Threshold':threshold})
    
            performance_conv = dict({'param':kernel_size,
                                    'AUC':auc_conv,
                                    'Threshold':conv_threshold})
    
            performance_median = dict({'param':kernel_size,
                                    'AUC':auc_median,
                                    'Threshold':median_threshold})
            
            performance_gauss = dict({'param':sigma,
                                    'AUC':auc_gauss,
                                    'Threshold':gauss_threshold})
            
            
    
            eval_data = dict({'gt_labels':gt_labels,
                              'scores':anomaly_scores,
                              'normal_scores':normal_scores})

            
            performance = {'standard':performance_stand,
                           'norm': performance_norm,
                           'conv': performance_conv,
                           'median': performance_median,
                           'gauss': performance_gauss}            
            
            spent_time = time.time() - start
                
            return performance, eval_data, spent_time
        
    def setLRscheduler(self, lr_scheduler_type=None, arg=None, epochs=None):
        if(lr_scheduler_type is None):
            self.lr_policy = None
            self.lr_scheduler_gen = torch.optim.lr_scheduler.StepLR(self.model.optimizer_gen,
                                                                    step_size=20, gamma=1)
            self.lr_scheduler_discr = torch.optim.lr_scheduler.StepLR(self.model.optimize_discr,
                                                                      step_size=20, gamma=1)
            
        elif(lr_scheduler_type == LR_DECAY and arg is not None and epochs is not None):
            print('LR SCHEDULING: Decay')
            self.lr_policy = LR_DECAY
            self.lr_scheduler_gen = torch.optim.lr_scheduler.StepLR(self.model.optimizer_gen,
                                                                    step_size=epochs, gamma=arg)
            self.lr_scheduler_discr = torch.optim.lr_scheduler.StepLR(self.model.optimizer_discr,
                                                                      step_size=epochs, gamma=arg)
        
        elif(lr_scheduler_type == LR_ONECYCLE, arg is not None and epochs is not None):
            print('LR SCHEDULING: OneCycle')
            self.lr_policy = LR_ONECYCLE
            self.lr_scheduler_gen = torch.optim.lr_scheduler.OneCycleLR(self.model.optimizer_gen, 
                                                                        steps_per_epoch=len(self.trainloader),
                                                                        max_lr=arg,
                                                                        epochs = epochs)
            self.lr_scheduler_discr = torch.optim.lr_scheduler.OneCycleLR(self.model.optimizer_discr, 
                                                                        steps_per_epoch=len(self.trainloader),
                                                                        max_lr=arg,
                                                                        epochs = epochs)
        else:
            raise Exception('LR type non VALID')
        
    def _training_step(self, epochs, one_cycle_maxLR=None, decay=None, save=True, lr_decay_value=None):
        
        plotUnit = 1
        start_epoch = self.epoch
        end_epochs = epochs
        
        self.lr_policy = None
        
#        if(decay is not None and one_cycle_maxLR is None) :
#            self.lr_policy = 'decay'
#            self.lr_scheduler_gen = torch.optim.lr_scheduler.StepLR(self.model.optimizer_gen,
#                                                                    step_size=20, gamma=decay)
#            self.lr_scheduler_discr = torch.optim.lr_scheduler.StepLR(self.model.optimizer_discr,
#                                                                      step_size=20, gamma=decay)
#        elif(decay is None and one_cycle_maxLR is not None):
#            self.lr_policy = 'oneCycle'
#            self.lr_scheduler_gen = torch.optim.lr_scheduler.OneCycleLR(self.model.optimizer_gen, 
#                                                                        steps_per_epoch=len(self.trainloader),
#                                                                        max_lr=one_cycle_maxLR,
#                                                                        epochs = epochs)
#            self.lr_scheduler_discr = torch.optim.lr_scheduler.OneCycleLR(self.model.optimizer_discr, 
#                                                                        steps_per_epoch=len(self.trainloader),
#                                                                        max_lr=one_cycle_maxLR,
#                                                                        epochs = epochs)
#        else:
#            self.lr_policy = None
#            self.lr_scheduler_gen = torch.optim.lr_scheduler.StepLR(self.model.optimizer_gen,
#                                                                    step_size=20, gamma=1)
#            self.lr_scheduler_discr = torch.optim.lr_scheduler.StepLR(self.model.optimizer_discr,
#                                                                      step_size=20, gamma=1)
            
        
        self.es = EarlyStopping(self.opt.patience)
        self.lrDecay = LR_decay(self.opt.lr_gen)
        
        for self.epoch in range(self.epoch, epochs):
            print('\n')
            print('Epoch {}/{}'.format(self.epoch+1, epochs))
            
            # TRAINING
            loss, losses, train_time = self._trainOneEpoch()

            self.train_loss[GENERATOR] = np.concatenate((self.train_loss[GENERATOR], [np.mean(loss[GENERATOR])]))
            self.train_loss[DISCRIMINATOR] = np.concatenate((self.train_loss[DISCRIMINATOR], [np.mean(loss[DISCRIMINATOR])]))
#            print(losses)
#            print(losses[0])
            
            self.train_adv_loss = np.concatenate((self.train_adv_loss, getNmeans(losses[0], plotUnit)))
            self.train_con_loss = np.concatenate((self.train_con_loss, getNmeans(losses[1], plotUnit)))
            self.train_enc_loss = np.concatenate((self.train_enc_loss, getNmeans(losses[2], plotUnit)))
            
#            train_loss[GENERATOR] = np.concatenate((train_loss[GENERATOR], loss[GENERATOR]))
#            print(len(train_loss[GENERATOR]))
            train_time = adjustTime(train_time)
            
            # VALIDATION
            loss, losses, val_time = self._validation()
#            print(loss[GENERATOR])
            self.val_loss[GENERATOR] = np.concatenate((self.val_loss[GENERATOR], [np.mean(loss[GENERATOR])]))
            self.val_loss[DISCRIMINATOR] = np.concatenate((self.val_loss[DISCRIMINATOR], [np.mean(loss[DISCRIMINATOR])]))
            
            self.valid_adv_loss = np.concatenate((self.valid_adv_loss, [np.mean(losses[0])]))
            self.valid_con_loss = np.concatenate((self.valid_con_loss, [np.mean(losses[1])]))
            self.valid_enc_loss = np.concatenate((self.valid_enc_loss, [np.mean(losses[2])]))
#            val_loss[GENERATOR] = np.concatenate((val_loss[GENERATOR], loss[GENERATOR]))
#            print(len(val_loss[GENERATOR]))
            val_time = adjustTime(val_time)

            valid_loss = self.val_loss['GENERATOR'][-1]
            
#            self.performance, eval_data, spent_time = self._test()
#            test_time = adjustTime(spent_time)
            
#            self.auc, self.threshold = self.performance['standard']['AUC'], self.performance['standard']['Threshold']           
#            self.gt_labels, self.anomaly_scores= eval_data['gt_labels'], eval_data['scores']
#            self.normal_scores = eval_data['normal_scores']
            
            if(self.epoch % 5 == 0):
                self.plotting()
#                self.evaluateRoc()
            
#            if(self.auc > self.best_auc['AUC']):
#                self.best_auc['AUC'] = self.auc
#                self.best_auc['Loss'] = valid_loss
#                self.best_auc['Threshold'] = self.threshold
                            
            print('\n')
            print('>- Training Loss:   {:.3f} in {} sec'.format(self.train_loss[GENERATOR][-1], train_time) )
            print('>- Validation Loss: {:.3f} in {} sec'.format(self.val_loss[GENERATOR][-1], val_time))
#            print('>- AUC: {:.3f} \n>- Threshold: {:.3f} in {} sec'.format(self.auc, self.threshold, test_time))            
            
            
            saveCkp = self.es(valid_loss)
            if(saveCkp and save):
#                print(valid_loss)
                self.saveCheckPoint(valid_loss)
                
            if(self.es.early_stop):
                print('-> Early stopping now')
                if(lr_decay_value):
                    print('-> LR factor decay: {}'.format(lr_decay_value))
                    self.lrDecay(lr_decay_value)
                    self.model.optimize_gen.params_groups[0]['lr'] = self.lrDecay.lr
                    print('New Learning rate for generator is {}'.format(self.lrDecay.lr))
                else:
                    break
        
#            print('Multi loss task weighting')
#            if(self.opt.multiTaskLoss):
#                print('\n')
#                print('Multi Task Losses\n')
#                steps = 30
#                mtl = MultiLossWrapper(self, self.trainloader, 3)
##                optimizer = torch.optim.Adam(mtl.multiTaskLoss.parameters(), lr=1e-03)
#                optimizer = torch.optim.SGD(mtl.multiTaskLoss.parameters(), lr=1e-01)
#                w_adv, w_con, w_enc = mtl.train(steps, optimizer)
#                self.model.setWeights(w_adv, w_con, w_enc)
        
        
        self.saveCheckPoint(valid_loss, last=True)
        self.plotting()
        self.saveInfo()
#        self.evaluateRoc(folder_save=self.folder_save)
        
        print('\n> From {} to {} epochs'.format(start_epoch, end_epochs))
        
        return {'Validation_Loss' : valid_loss,
#                'AUC': self.auc,
#                'Threshold' : self.threshold
                }
        
        
    def train_model(self, epochs, save=True, lr_decay_value=None):
        
        print('-> Training model: ', self.opt.name)
        self.model.generator.train()
        self.model.discriminator.train()
        
        self.epoch = 0
        
        self.train_loss = {GENERATOR:[], DISCRIMINATOR:[]}
        
        self.val_loss = {GENERATOR:[], DISCRIMINATOR:[]}

        self.train_adv_loss = []
        self.train_con_loss = []
        self.train_enc_loss = []
        
        self.valid_adv_loss = []
        self.valid_con_loss = []
        self.valid_enc_loss = []
        
        self.folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(self.folder_save)
        
        self.results_folder = paths.checkpoint_folder + self.opt.name + '/' + self.opt.name + '_training_result/'
        ensure_folder(self.results_folder)
        
        self.best_auc = {'AUC':0, 'Loss':0, 'Threshold':0}

        performance = self._training_step(epochs, save, lr_decay_value)
        
        return performance
    
    
    def resumeTraining(self, epochs, save=True, one_cycle_maxLR=None, decay=None):
        
        performance = self._training_step(epochs, one_cycle_maxLR, decay, save)
        
        return performance
    
    def plotting(self, save=True):
        
        # PLOTTING LOSSES
        fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(5,1, figsize=(8,16))
        
        _subplot(ax1, self.train_adv_loss, self.valid_adv_loss, 'ADV loss')
        _subplot(ax2, self.train_con_loss, self.valid_con_loss, 'CON loss')
        _subplot(ax3, self.train_enc_loss, self.valid_enc_loss, 'ENC loss')
        _subplot(ax4, self.train_loss[GENERATOR], self.val_loss[GENERATOR], 'Generator')
        _subplot(ax5, self.train_loss[DISCRIMINATOR], self.val_loss[DISCRIMINATOR], 'Discriminator')
#        plt.legend()        
        
        if(save):
#            plt.savefig(self.folder_save + self.opt.name + '/'+ 'plot')
            plt.savefig(self.results_folder + 'plot')
       
        plt.show()
    
    def evaluateRoc(self, mode='standard', param=None,
                    folder_save=None, plot=True):
        
        if(folder_save is not None):
            folder_save = folder_save
        
        if(mode == 'conv'):
            assert param is not None, 'kernel size NONE'
            scores = convFilterScores(self.anomaly_scores, param)
        
        elif(mode == 'median'):
            assert param is not None, 'kernel size NONE'
            scores = medFilterScores(self.anomaly_scores, param)
        
        elif(mode == 'gauss'):
            assert param is not None, 'Wrong gauss params EVALUATE ROC'
            scores = gaussFilterScores(self.anomaly_scores, param)
        
        elif(mode == 'standard'):
            scores = self.anomaly_scores
        
        auc, thr = evaluate(self.gt_labels, scores, plot=plot, folder_save=folder_save)
        
        print('\n')
        print('AUC: {:.3f} \t Thres. : {:.6f} '.format(auc, thr))
        
        return auc, thr
    
    def saveInfo(self):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        saveInfoGanomaly(self.opt, folder_save, self.auc)
    
    def addInfo(self, info):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        addInfoGanomaly(self.opt, folder_save, info)
    
    
    def predict(self, image, threshold=None, target=None, info=None, verbose=0):
#        image_tensor = torch.FloatTensor(image)
        image_transf = Transforms.ToTensor()(image)
        image_unsqueeze = image_transf.unsqueeze_(0)
        x_image = Variable(image_unsqueeze).cuda()
        
        transf = Transforms.Compose([Transforms.ToTensor(), Transforms.Normalize((0.5,),(0.5,))])
        
        imagePIL = Image.fromarray(image)
        x = transf(imagePIL)
        x = x.unsqueeze_(0)
        x = Variable(x).cuda()
#        image = image_tensor
        
#        plt.imshow(image)

        with torch.no_grad():
            _, z, z_prime = self.model.forward_gen(x)
            x_prime, _, _ = self.model.forward_gen(x_image)

        score = torch.mean(torch.pow((z-z_prime), 2), dim=1)
#        print(score)
        anomaly_score = score
        
        output = x_prime.cpu().numpy()
        final_output = np.transpose(output[0], (2,1,0))
        
#        final_output = (output * 0.5) + 0.5
        final_output = np.flip(final_output, 1)
        final_output = np.rot90(final_output, 1)        
        
        
        if(threshold is not None):
            thr = threshold
        else:
            thr = self.threshold
        
        prediction = ['Anomalous Image', 1] if score >= thr else ['Normal Image', 0]
            
        if(target is not None):
            real_outcome = 'Anomalous Image' if target == 1 else 'Normal Image'        

        if(verbose):
            
            fig, [ax1, ax2] = plt.subplots(2,1, figsize=(10,13))
            results = '\n------------ RESULTS -------------\n' + \
                       'Threshold: {:.3f}\n'.format(thr) + \
                       'Score: {:.3f}\n'.format(anomaly_score.item()) + \
                       'Real Outcome: {}\n'.format(real_outcome) + \
                       '---------------------------------\n\n' + \
                       'Original image --> {}'.format(prediction[0])
                       
            ax1.set_title(results)
            ax1.imshow(image)
            ax2.set_title('Reconstructed image')
            ax2.imshow(final_output)
            
            print('')
            print('\n------------ RESULTS -------------')
            print('Threshold: \t{:.3f}'.format(thr))
            print('Score: \t\t{:.3f}'.format(anomaly_score.item()))
            print('From \t\t{}'.format(real_outcome))
            print('')
            print('Original image --> ', prediction[0])
            print('----------------------------------')
            
            if(info is not None):
                print('..Saving..')
                if(prediction[0] == 'Normal Image'):    
                    plt.savefig(self.folder_save + 'Normal_{}'.format(info))
                elif(prediction[0] == 'Anomalous Image'):
                    plt.savefig(self.folder_save + 'Anomaly_{}'.format(info))
                else:
                    raise Exception('Wrong Predicion')
            
        return prediction, anomaly_score.item(), thr
        
    
    def predictImage(self, dataTest, folder_save=None, N=10):
        
        i = 0
        
        with torch.no_grad():
            anomaly_scores = torch.zeros(size=(len(dataTest.dataset),), dtype=torch.float32, device=device)
            gt_labels = torch.zeros(size=(len(dataTest.dataset),), dtype=torch.long, device=device)
                
            start = time.time()
            for patches, labels in tqdm(dataTest, total=len(dataTest)):
                
                x = torch.Tensor(patches).cuda()
                tensor_label = torch.Tensor(labels).cuda()
                
                
                x_prime, z, z_prime = self.model.forward_gen(x)
                
                # ANOMALY SCORE
                score = torch.mean(torch.pow((z-z_prime), 2), dim=1)
                    
                anomaly_scores[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
                gt_labels[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = tensor_label.reshape(score.size(0))
        
                i += 1
    
            anomaly_scores_norm = (anomaly_scores - torch.min(anomaly_scores)) / (torch.max(anomaly_scores) - torch.min(anomaly_scores))
            auc, threshold_auc = evaluate(gt_labels, anomaly_scores_norm, plot=True, folder_save=folder_save)
            
            avg_prec = evaluate(gt_labels, anomaly_scores_norm, metric='prec_rec_curve', plot=True, folder_save=folder_save)
#            return evaluate(gt_labels, anomaly_scores_norm, metric='prec_rec_curve', plot=True)
            end = time.time()
            
            performance = {'AUC': auc,
                           'Thr': threshold_auc,
                           'Avg_prec': avg_prec}
            
            print('Prediction time: {}'.format(adjustTime(end-start)))
            
            gt_labels = dataTest.dataset.targets
            pred_labels = computeAnomalyDetection(anomaly_scores_norm, threshold_auc)
            # PRECISION
            precision = evaluate(gt_labels, pred_labels, metric='precision')
            
            
            pred_patches = generatePatches(dataTest.dataset.data, pred_labels)
            
            
            print('Precision: {}'.format(precision))
            samples = getSamples(patches, x_prime, labels, anomaly_scores_norm, N=10)
            
        return pred_patches, samples, performance
            
    def saveImages(self, dataloader):
        
        reals, fakes, fixed = self.get_images(dataloader)
        self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
        if self.opt.display:
            self.visualizer.display_current_images(reals, fakes, fixed)
    
        
    def saveCheckPoint(self, valid_loss, last=False):
        self.folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(self.folder_save)
#        print(valid_loss)
        path_file = '{0}/MODEL_{1}_lr:{2}|Epoch:{3}|Loss:{4:.4f}.pth.tar'.format(self.folder_save,
                                                                             self.opt.name,
                                                                             self.opt.lr_gen,
                                                                             self.epoch,
                                                                             valid_loss)
        
        torch.save(self, path_file)
        
        if(last == False):
            path_best_ckp = '{0}/{1}_best_ckp.pth.tar'.format(self.folder_save, self.opt.name)
            torch.save(self, path_best_ckp)
        
        
        # SAVE JUST A CHECKPOINT
#        ckp = Checkpoint(self)
#        ckp.saveCheckpoint(valid_loss)
        
    def tuneLearningRate(self, inf_bound_gen, sup_bound_gen, inf_bound_discr, sup_bound_discr):
        
        max_count = 10
        self.result = []
        
        for count in range(max_count):
            
            print('Model n.', count)
            
            self.opt.epochs = 4
            self.model.optimizer_gen.lr = 10**np.random.uniform(sup_bound_gen, inf_bound_gen)
            self.model.optimizer_discr.lr = 10**np.random.uniform(sup_bound_discr, inf_bound_discr)
            loss = self.train_model(save=False)
            
            lr_gen_label = 'Gen_Lr:\t{}\n'.format(self.model.optimizer_gen.lr)
            lr_discr_label = 'Discr_Lr:\t{}\n'.format(self.model.optimizer_discr.lr)
            loss_label = 'Loss:\t{}\n\n'.format(loss)
            
            result_label = 'Results \n' + lr_gen_label + lr_discr_label + loss_label
            
            self.result.append(result_label)
        
        return self.result

        

def computeAnomalyDetection(scores, threshold):   

    pred_labels = []
    
    for i in range(0, len(scores)):
        if(scores[i] < threshold):
            pred_labels.append(np.float64(0))
        else:
            pred_labels.append(np.float64(1))
            
    return pred_labels
    
def outputSample(sample, threshold, info=None, folder_save=None):
    
    original = sample['originals']
    image_rec = sample['images']
    label = 'Anomaly' if(sample['labels']) else 'Normal'
    score = sample['scores']
    print(score)
    
    result = 'Normal patch' if score < threshold else 'Anomalous patch'
    
    fig, [ax1, ax2] = plt.subplots(2,1, figsize=(10,13))
    results = '\n------------ RESULTS -------------\n' + \
               'Threshold: {:.3f}\n'.format(threshold) + \
               'Score: {:.3f}\n'.format(score) + \
               'Real Outcome: {}\n'.format(label) + \
               '---------------------------------\n\n' + \
               'Original image --> {}'.format(result)
               
    ax1.set_title(results)
    ax1.imshow(original)
    ax1.grid(False)
    ax2.set_title('Reconstructed image')
    ax2.imshow(image_rec)
    ax2.grid(False)
    
    print('')
    print('\n------------ RESULTS -------------')
    print('Threshold: \t{:.3f}'.format(threshold))
    print('Score: \t\t{:.3f}'.format(score))
    print('From \t\t{}'.format(label))
    print('')
    print('Original image --> ', result)
    print('----------------------------------')
    
    if(info is not None and folder_save is not None):
        print('..Saving..')
        if(result == 'Normal patch'):    
            plt.savefig(folder_save + 'Normal_{}'.format(info))
        elif(result == 'Anomalous patch'):
            plt.savefig(folder_save + 'Anomaly_{}'.format(info))
        else:
            raise Exception('Wrong Predicion')  
    
    
def _subplot(ax, train, val, title):
    ax.set_title(title)
    ax.plot(train, color='r', label='Training')
    ax.plot(val, color='b', label='Validation')
    ax.legend()
        
def adjustTime(sample_time):
    
    minutes = sample_time / 60
    seconds = (minutes - (sample_time // 60)) * 60
    
    return '{} min {} sec'.format(int(minutes), int(seconds))
      
def getSamples(originals, patches, labels, scores, N=10):
    
    patch_image = []
    patch_recon = []
    patch_labels = []
    patch_scores = []
    
    for i in range(0, N):
        x = originals[i]
        x_prime = patches[i]
        label = labels[i]
        
        image = x.cpu().numpy()
        output = x_prime.cpu().numpy()
#        print(output.shape)
        image_input = np.transpose(image, (2,1,0))
        final_output = np.transpose(output, (2,1,0))
#        print(final_output.shape)
        
        image_input = (image_input * 0.5) + 0.5
        image_input = np.flip(image_input, 1)
        image_input = np.rot90(image_input, 1) 
        
        final_output = (final_output * 0.5) + 0.5
        final_output = np.flip(final_output, 1)
        final_output = np.rot90(final_output, 1) 
        
        start = len(scores) - len(patches)
#        print(start)
        score = scores[start + i]
        
        patch_image.append(image_input)
        patch_recon.append(final_output)
        patch_labels.append(label.cpu())
        patch_scores.append(score)
        
        
    samples = {'originals': patch_image,
               'images': patch_recon,
               'labels': patch_labels,
               'scores': patch_scores}
    
    return pd.DataFrame(samples)
        
        
        
        
        
        
        
        
        
