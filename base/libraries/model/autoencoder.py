# -*- coding: utf-8 -*-

#%% IMPORTS
import time
from collections import OrderedDict

from libraries.model.network import Encoder, Decoder, weights_init
from libraries.model.loss import contextual_loss
from libraries.model.evaluate import evaluate
from libraries.utils import Paths
from libraries.utils import Checkpoint, saveInfoAE, addInfoAE
from libraries.utils import ensure_folder
from libraries.utils import EarlyStopping
from libraries.model.postprocessing import convFilterScores, medFilterScores, gaussFilterScores

import torch
import torch.nn as nn
from torchvision import transforms as Transforms

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

from tqdm import tqdm
import sys
from torch.autograd import Variable
#import sys

paths = Paths()
#%% NETWORK

device = torch.device('cuda:0')

class Autoencoder(nn.Module):
    
    def __init__(self, opt):
        super().__init__()
        
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, x):
        
        z = self.encoder(x)
        x_prime = self.decoder(z)
        
        return x_prime

def generateAutoencoder(opt, cuda=True):
    
    if(cuda):
        return Autoencoder(opt).cuda()        
    else:
        return Autoencoder(opt).cpu()

def loadAEmodel(path_file):
    return torch.load(path_file)

class AutoencoderModel():
    
    def __init__(self, opt, optimizer, trainloader=None, validationloader=None, testloader=None):
        
        self.model      = generateAutoencoder(opt)
        self.optimizer  = optimizer(self.model.parameters(), opt.lr)
        self.loss_function = contextual_loss()
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.testloader = testloader
        self.opt = opt
        
        self.model.apply(weights_init)
    
#    def loadCheckPoint(self, path_file):
#        self.model = torch.load(path_file)
#        self.optimizer = checkpoint.optimizer
    
    def loadTrainloader(self, trainloader):
        self.trainloader = trainloader
        
    def loadValidationLoader(self, validationloader):
        self.validationloader = validationloader
     
    def _trainOneEpoch(self):
    
        trainLoss = []
        
        self.model.train()
        
        start = time.time()
        
        n = len(self.trainloader)
        for images, labels in tqdm(self.trainloader, leave=True, position=0,
                                   total=n, desc='Training\t',
                                   file = sys.stdout):
            
            x = torch.Tensor(images).cuda()
            
            # FORWARD
            output = self.model(x)        
            
            loss = self.loss_function(output, x)
                    
            # BACKWARD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # VISUALIZATION
            trainLoss.append(loss.item()*images.size(0))
                  
        time_spent = time.time() - start
            
        return trainLoss, time_spent
        
    def _validation(self):
    
        validationLoss = []
    
        self.model.eval()      
        
        start = time.time()
        
        with torch.no_grad():
            count=0
            self.curr_steps += self.batch_size
            
            n = len(self.validationloader)
            
            for images, labels in tqdm(self.validationloader, leave=True, position=0, total=n,
                                     file = sys.stdout, desc='Validation\t'):
                
                count += 1
                
                x = torch.Tensor(images).cuda()
                
                # FORWARD
                output = self.model(x)       
                        
                loss = self.loss_function(output, x)
                
                # VISUALIZATION
                validationLoss.append(loss.item()*images.size(0))
            
            time_spent = time.time() - start
            
        return validationLoss, time_spent
    
    def _test(self):
        
        start = time.time()
        
        with torch.no_grad():
            
            i=0
            curr_epoch = 0
            times = []
            n_iter = len(self.validationloader)

            anomaly_scores = torch.zeros(size=(len(self.testloader.dataset),), dtype=torch.float32, device=device)
            gt_labels = torch.zeros(size=(len(self.testloader.dataset),), dtype=torch.long,    device=device)
            
            for images, labels in tqdm(self.testloader, leave=True, total=n_iter, desc='Test\t\t', file = sys.stdout):
                
                curr_epoch += self.opt.batch_size
                
                time_in = time.time()
                
                x = torch.Tensor(images).cuda()
                tensor_labels = torch.Tensor(labels).cuda()                
                
                x_prime = self.model(x)
                
                # ANOMALY SCORE
                x = x.reshape(x.size(0), x.size(1)*x.size(2)*x.size(3))
                x_prime = x_prime.reshape(x_prime.size(0), x_prime.size(1)*x_prime.size(2)*x_prime.size(3))
                
                score = torch.mean(torch.pow((x-x_prime), 2), dim=1)
                
                time_out = time.time()
                
                anomaly_scores[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
                gt_labels[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = tensor_labels.reshape(score.size(0))
              
                times.append(time_out - time_in)

                i += 1
            
            
            # NORMALIZATION - Scale error vector between [0, 1]
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
                              'scores':anomaly_scores})

            
            performance = {'standard':performance_stand,
                           'norm': performance_norm,
                           'conv': performance_conv,
                           'median': performance_median,
                           'gauss': performance_gauss}   
    
            spent_time = time.time() - start
                
            return performance, eval_data,  spent_time
    
    def _trainingStep(self, epochs, save):
        
        for self.epoch in range(epochs):
            print('\n')
            print('Epoch {}/{}'.format(self.epoch+1, epochs))
            
            # TRAINING
            train_losses, train_time = self._trainOneEpoch()
            train_loss = np.average(train_losses)
#            self.loss['train'].append(train_losses) 
            self.avg_loss['train'].append(train_loss)
            
            # VALIDATION
            val_losses, val_time     = self._validation()
            val_loss = np.average(val_losses)
#            self.loss['validation'].append(val_losses)
            self.avg_loss['validation'].append(val_loss)
            
            # TEST
            self.performance, eval_data, spent_time = self._test()
            
            self.auc, self.threshold = self.performance['standard']['AUC'], self.performance['standard']['Threshold']           
            self.gt_labels, self.anomaly_scores = eval_data['gt_labels'], eval_data['scores']            
            
            if(self.epoch % 5 == 0):
                self.plotting(save=True)
                self.evaluateRoc()
            
            print('\n')
            print('>- Training Loss:   {:.4f} in {:.2f} sec'.format(train_loss, train_time) )
            print('>- Validation Loss: {:.4f} in {:.2f} sec'.format(val_loss, val_time))
            
            # SAVING CHECKPOINT
            
            ensure_folder(self.folder_save)
            
#            filename = '{0}/{1}_lr:{2}|Epoch:{3}|Loss:{4:.3f}.pth.tar'.format(folder_save,
#                                                                                self.opt.name,
#                                                                                self.opt.lr,
#                                                                                epoch, train_loss)
            
            saveCkp = self.es(val_loss)
            if(saveCkp and save):
                self.saveCheckPoint(val_loss)
            
            if(self.es.early_stop):
                print('-> Early stopping now')
#                self.plotting()
                break
        
        self.saveCheckPoint(val_loss)
        self.plotting()
        self.evaluateRoc(folder_save=self.folder_save)
        self.saveInfo()
        
        return {'Validation_Loss' : val_loss,
                'AUC': self.auc,
                'Threshold' : self.threshold}
    
    def train_model(self, epochs, save=True):
        
        self.curr_steps = 0
        self.batch_size = self.opt.batch_size
        
        self.es = EarlyStopping(self.opt)
        
        self.loss = {}
        self.loss['train'] = []
        self.loss['validation'] = []
        
        self.avg_loss = {}
        self.avg_loss['train'] = []
        self.avg_loss['validation'] = []
        
        self.folder_save = paths.checkpoint_folder
        self.results_folder = paths.checkpoint_folder + self.opt.name + '/' + self.opt.name + '_training_result/'
        ensure_folder(self.results_folder)
        
        assert self.trainloader is not None, 'None Trainloader'
        assert self.validationloader is not None, 'None Validloader'
        assert self.testloader is not None, 'None Testloader'
        
        performance = self._trainingStep(epochs, save)
        
        return performance
    
    def resumeTraining(self, epochs, save=True):
        
        performance = self._training_step(epochs, save)
        
        return performance
    
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
        
        saveInfoAE(self.opt, folder_save, self.auc)
    
    def addInfo(self, info):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        addInfoAE(self.opt, folder_save, info)
        
    
    def saveCheckPoint(self, val_loss):
        
        self.folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(self.folder_save)
        
        path_file = '{0}/{1}_lr:{2}|Epoch:{3}|Loss:{4:.4f}.pth.tar'.format(self.folder_save,
                                                                             self.opt.name,
                                                                             self.opt.lr,
                                                                             self.epoch,
                                                                             val_loss)
        
        torch.save(self, path_file)
     
    def plotting(self, save=True):
                
        # PLOTTING LOSSES
        plt.figure()
        plt.title('Average Loss')
        plt.plot(self.avg_loss['train'], color='r', label='train')
        plt.plot(self.avg_loss['validation'], color='b', label='validation')
        plt.legend()
        
        
        if(save):
#            plt.savefig(self.folder_save + self.opt.name + '/'+ 'plot')
            plt.savefig(self.folder_save + 'plot')
       
        plt.show()
        
    def plotAnomScores(self):
        anom_scores = []
        for elem in self.anomaly_scores:
            anom_scores.append(elem.item())
        
        plt.hist(anom_scores)
        
    def predict(self, image, target=None, threshold=None, 
                info=None, verbose=0):
        
        if(self.opt.in_channels == 1):
            cmap = 'gray'
            grayMode = True
            
        elif(self.opt.in_channels == 3):
            cmap = None
            grayMode = False
        
        image_transf = Transforms.ToTensor()(image)
        image_unsqueeze = image_transf.unsqueeze_(0)
        x = Variable(image_unsqueeze).cuda()

        with torch.no_grad():
            x_prime = self.model(x)
            
        # SCORE
        
        #torch.Size([1, 3, 32, 32])
        
        x_score = x.reshape(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x_prime_score = x_prime.reshape(x_prime.size(0), x_prime.size(1)*x_prime.size(2)*x_prime.size(3))
        
        score = torch.mean(torch.pow((x_score-x_prime_score), 2), dim=1)
        
        # NORM SCORE
#        anomaly_score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
        anomaly_score = score
        
        output = x_prime.cpu().numpy()
        output = np.transpose(output[0], (2,1,0))

        if(grayMode):
            output = output[0].reshape(self.opt.img_size, self.opt.img_size)

        final_output = output
#        final_output = (output * 0.5) / 0.5
        final_output = np.flip(final_output, 1)
        final_output = np.rot90(final_output, 1)
        
#        plt.imshow(image, cmap='gray')
#        plt.show()
#        plt.imshow(image)
        
        if(threshold is not None):
            thr = threshold
        else:
            thr = self.threshold
        
        prediction = ['Anomalous Image', 1] if score >= thr else ['Normal Image', 0]
#        print(prediction)
        if(verbose):
            
            fig, [ax1, ax2] = plt.subplots(2,1, figsize=(10,10))
            results = '------------ RESULTS -------------\n' + \
                       'Threshold: {:.3f}\n'.format(thr) + \
                       'Score: {:.3f}\n'.format(anomaly_score.item()) + \
                       '---------------------------------\n\n' + \
                       'Original image --> {}'.format(prediction[0])
                       
            ax1.set_title(results)
            ax1.imshow(image, cmap=cmap)
            ax2.set_title('Reconstructed image')
            ax2.imshow(final_output, cmap=cmap)
            
            print('')
            print('\n------------ RESULTS -------------')
            print('Threshold: \t{:.3f}'.format(thr))
            print('Score: \t\t{:.3f}'.format(anomaly_score.item()))
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
        
    def tuneLearningRate(self, inf_bound, sup_bound):
        
        max_count = 10
        result = []
        
        for count in range(max_count):
            
            print('Model n.', count)
            
            self.opt.epochs = 5
            self.optimizer.lr = 10**np.random.uniform(sup_bound, inf_bound)
            loss = self.train_autoencoder(save=False)
            
            result.append(str(self.optimizer.lr) + ' : ' + str(loss))
        
        return result
                       
        
        
        
        











