import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import random

from loss import Classifying_Consensus_Loss
import evaluation
from utils import next_batch,tensor_shuffle,drop_feature
import math
import ipdb
import time as Time



class Autoencoder(nn.Module):
    def __init__(self, encoder_dim, activation='relu', batchnorm=True, AE_proj=False):
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm
        self.L_or_S = AE_proj

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        # encoder_layers.append(nn.Softmax(dim = 1))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        if self.L_or_S:
            decoder_layers.append(nn.Softmax(dim = 1))
        self.decoder = nn.Sequential(*decoder_layers)


        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight.data)
        #     # torch.nn.init.xavier_normal_(m.weight.data)
        #     # torch.nn.init.uniform_(m.weight.data)
        #     # torch.nn.init.normal_(m.weight.data)
        # torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        # torch.nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))



    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # torch.nn.init.xavier_uniform_(m.weight.data)
                # torch.nn.init.xavier_normal_(m.weight.data)
                # torch.nn.init.uniform_(m.weight.data)
                # torch.nn.init.normal_(m.weight.data)
                torch.nn.init.kaiming_uniform_(m.weight, a = math.sqrt(5))
                # torch.nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class mvc():
    def __init__(self,args, config):
        self._config = config
        self._args = args

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')


        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder3 = Autoencoder(config['Autoencoder']['arch3'], config['Autoencoder']['activations3'],
                                        config['Autoencoder']['batchnorm'])

        self.sm = nn.Softmax(dim = 1)

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.cuda()
        self.autoencoder2.cuda()
        self.autoencoder3.cuda()


    def model_init(self):
        self.autoencoder1.weights_init()
        self.autoencoder2.weights_init()
        self.autoencoder3.weights_init()

        # self.view1_to_view2.weights_init()
        # self.view2_to_view1.weights_init()

        # self.autoencoder1.apply(weights_init)
        # self.autoencoder2.apply(weights_init)
        # self.view1_to_view2.appply(weights_init)
        # self.view2_to_view1.apply(weights_init)

    
    def train(self, config, logger, XX1, XX2, XX3,gt_label, optimizer, device):
        # ipdb.set_trace()
        scores = {}
        scores['kmeans']={'AMI': 0, 'NMI': 0, 'ARI': 0,'accuracy': 0, 'precision': 0, 'recall': 0, 'f_measure': 0}
        best_acc = 0
        for epoch in range(config['training']['epoch']):
            X1, X2, X3 = shuffle(XX1, XX2,XX3)
            loss_all, loss_cls, loss_glb, loss_code,loss_rec = 0, 0, 0, 0, 0
            for x1, x2, x3,batch_No in next_batch(X1, X2, X3,config['training']['batch_size']):
                
                # Data augmentation
                x1_aug = drop_feature(x1, config['training']['droprate'])
                x2_aug = drop_feature(x2, config['training']['droprate'])
                x3_aug = drop_feature(x3, config['training']['droprate'])

                Z1 = self.autoencoder1.encoder(x1)
                z1 = self.sm(Z1)
                Z2 = self.autoencoder2.encoder(x2)
                z2 = self.sm(Z2)
                Z3 = self.autoencoder3.encoder(x3)
                z3 = self.sm(Z3)

                Z1_aug = self.autoencoder1.encoder(x1_aug)
                z1_aug = self.sm(Z1_aug)
                Z2_aug = self.autoencoder2.encoder(x2_aug)
                z2_aug = self.sm(Z2_aug)
                Z3_aug = self.autoencoder3.encoder(x3_aug)
                z3_aug = self.sm(Z3_aug)

                ### Reconstruction Objective 
                L1_rec = F.mse_loss(self.autoencoder1.decoder(Z1), x1)
                L2_rec = F.mse_loss(self.autoencoder2.decoder(Z2), x2)
                L3_rec = F.mse_loss(self.autoencoder3.decoder(Z3), x3)
                L4_rec = F.mse_loss(self.autoencoder1.decoder(Z1_aug), x1_aug)
                L5_rec = F.mse_loss(self.autoencoder2.decoder(Z2_aug), x2_aug)
                L6_rec = F.mse_loss(self.autoencoder3.decoder(Z3_aug), x3_aug)
 
                L_rec = L1_rec + L2_rec + L3_rec + L4_rec + L5_rec + L6_rec


                ### Classifying Consensus Learning
                L_cls1 = Classifying_Consensus_Loss(z1_aug, z2_aug, self._args,
                                                    config['training']['alpha'],
                                                    config['training']['beta'],
                                                    config['training']['gamma'])
                L_cls2 = Classifying_Consensus_Loss(z2_aug, z3_aug, self._args,
                                                    config['training']['alpha'],
                                                    config['training']['beta'],
                                                    config['training']['gamma'])
                L_cls3 = Classifying_Consensus_Loss(z1_aug, z3_aug, self._args,
                                                    config['training']['alpha'],
                                                    config['training']['beta'],
                                                    config['training']['gamma'])
                

                L_cls = L_cls1 + L_cls2 + L_cls3


                ### Global Consensus Learning
                L_glb1 = -(F.normalize(Z1, dim = 1)*F.normalize(Z2, dim = 1)).sum(dim = 1).mean()-(F.normalize(Z1_aug, dim = 1)*F.normalize(Z2_aug, dim = 1)).sum(dim = 1).mean()
                L_glb2 = -(F.normalize(Z2, dim = 1)*F.normalize(Z3, dim = 1)).sum(dim = 1).mean()-(F.normalize(Z2_aug, dim = 1)*F.normalize(Z3_aug, dim = 1)).sum(dim = 1).mean()
                L_glb3 = -(F.normalize(Z1, dim = 1)*F.normalize(Z3, dim = 1)).sum(dim = 1).mean()-(F.normalize(Z1_aug, dim = 1)*F.normalize(Z3_aug, dim = 1)).sum(dim = 1).mean()
                L_glb = L_glb1 + L_glb2 + L_glb3 




                ### Coding Consensus Learning
                max_probs1, targets_u1 = torch.max(z1.detach(), dim = -1)
                max_probs2, targets_u2 = torch.max(z2.detach(), dim = -1)
                max_probs3, targets_u3 = torch.max(z3.detach(), dim = -1)

                L_code1 = (F.cross_entropy(z1_aug, targets_u1, reduction='none') ).mean()
                L_code2 = (F.cross_entropy(z2_aug, targets_u2, reduction='none') ).mean()
                L_code3 = (F.cross_entropy(z3_aug, targets_u3, reduction='none') ).mean()
                L_code = L_code1 + L_code2 + L_code3


                ###  Overall loss function
                loss = L_cls + L_glb * config['training']['lambda1'] + L_code * config['training']['lambda2'] + L_rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_rec += L_rec.item()
                loss_glb += L_glb.item()
                loss_cls += L_cls.item()
                # if x1_aug2.numel():
                loss_code += L_code.item()


            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction Loss = {:.4f}  ===> Global Consensus Learning Loss = {:.4f} " \
                        "===> Classifying Consensus Learning Loss = {:.4e} ===> Coding Consensus  Learning Loss = {:.4f} ===> Total Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_glb, loss_cls, loss_code,loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")


            # evalution
            if (epoch + 1) % config['print_num'] == 0:
                
                scores = self.evaluation(config, logger, XX1, XX2,XX3, gt_label, device)

            # Best model and best result
            if scores['kmeans']['accuracy'] > best_acc:
                torch.save({'autoencoder1': self.autoencoder1.state_dict(),'autoencoder2':self.autoencoder2.state_dict(),'autoencoder3':self.autoencoder3.state_dict()}, './model/'+self._config['dataset']+'_best.pth')
                best_acc = scores['kmeans']['accuracy']
                best_t = epoch
                best_scores = scores            

        logger.info("\033[2;29m" +'best scores' + 'view_concat ' + str(best_scores) + "\033[0m")

        return best_scores['kmeans']['accuracy'], best_scores['kmeans']['NMI'], best_scores['kmeans']['ARI']


    
    def evaluation(self, config, logger, X1, X2,X3, gt_label, device):
        with torch.no_grad():

            self.autoencoder1.eval(), self.autoencoder2.eval()
            # representations
            view1_latent_eval = self.autoencoder1.encoder(X1).cuda()
            view2_latent_eval = self.autoencoder2.encoder(X2).cuda()
            view3_latent_eval = self.autoencoder3.encoder(X3).cuda()
            
            latent_code_view1_eval = torch.zeros(X1.shape[0], config['Autoencoder']['arch1'][-1]).cuda()
            latent_code_view2_eval = torch.zeros(X2.shape[0], config['Autoencoder']['arch2'][-1]).cuda()
            latent_code_view3_eval = torch.zeros(X3.shape[0], config['Autoencoder']['arch2'][-1]).cuda()

            latent_code_view1_eval = view1_latent_eval
            latent_code_view2_eval = view2_latent_eval
            latent_code_view3_eval = view3_latent_eval

            # Fuse feature by concatenate
            latent_fusion = torch.cat([latent_code_view1_eval, latent_code_view2_eval,latent_code_view3_eval], dim = 1).cpu().numpy()
            # k-means clustering
            scores = evaluation.clustering([latent_fusion], gt_label[0])
            logger.info("\033[2;29m" + 'view_concat ' + str(scores) + "\033[0m")

            self.autoencoder1.train(), self.autoencoder2.train()

        return scores