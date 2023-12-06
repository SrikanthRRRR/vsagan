"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lib.models.networks import NetD, weights_init, define_G, define_D, get_scheduler
from lib.visualizer import Visualizer
from lib.loss import l2_loss, var_loss
from lib.evaluate import roc
from lib.models.basemodel import BaseModel
from lib.models.skipganomaly import Skipganomaly


class Vsagan(Skipganomaly):
    """GANomaly Class
    """
    @property
    def name(self): return 'Vsagan'

    def __init__(self, opt, data=None):
        super(Vsagan, self).__init__(opt, data)

        ##
        # Loss Functions
        self.l_var = var_loss

    def backward_g(self):
        """ Backpropagate netg
        """
        
        #print("Vsagan_backward_g")

        self.err_g_adv = self.opt.w_adv * self.l_adv(self.pred_fake, self.real_label)
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)
        self.err_g_lat = self.opt.w_lat * self.l_lat(self.feat_fake, self.feat_real)

        if self.opt.g_lat_dim > 0:
            self.err_g_var = self.opt.w_var * self.l_var(self.pred_real, self.pred_fake)

            self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat + self.err_g_var
        else:
            self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat

        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        #print("vsagan_backward_d")
        # Fake
        pred_fake, _ = self.netd(self.fake.detach())
        self.err_d_fake = self.l_adv(pred_fake, self.fake_label)

        # Real
        # pred_real, feat_real = self.netd(self.input)
        self.err_d_real = self.l_adv(self.pred_real, self.real_label)

        # Combine losses.
        self.err_d = self.err_d_real + self.err_d_fake
        self.err_d.backward(retain_graph=True)

        ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        #print("vsagan_get_errors")
        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_lat', self.err_g_lat.item())])

        if self.opt.g_lat_dim > 0:
            errors['err_g_var'] = self.err_g_var.item()

        return errors

        ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0
        best_auc_since = 0
        i = 0
        # Train for niter epochs.
        
        print(f">> Training {self.name} on {self.opt.dataset} to detect {self.opt.abnormal_class}")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_one_epoch()
            res = self.test()
            i+=1

            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch)
                #print(self.epoch)
                best_auc_since = 0
                self.best_auc = best_auc
                self.best_auc_epoch = self.epoch
            else:
                best_auc_since+=1

            print(self.get_errors())
            self.visualizer.print_current_performance(res, best_auc)
            if(best_auc_since == self.opt.stopiternoimp):
                break

        if(self.opt.saveBestModelOutput):
            #print(self.best_auc_epoch)
            self.load_weights(self.best_auc_epoch)
            for i, data in enumerate(self.data.valid, 0):
                if i == 1 or i == 17:
                    with torch.no_grad():
                        if i == 1:
                            self.set_input(data)
                            self.fake = self.netg(self.input)
                            reals, fakes, fixed = self.get_current_images()
                            self.visualizer.save_current_images(self.best_auc_epoch, reals, fakes, fixed, "normal")

                        if i == 17:
                            self.set_input(data)
                            self.fake = self.netg(self.input)
                            reals, fakes, fixed = self.get_current_images()
                            self.visualizer.save_current_images(self.best_auc_epoch, reals, fakes, fixed, "abnormal")

        print(">> Training model %s.[Done]" % self.name)

        ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """
        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.data.train, leave=False, total=len(self.data.train)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.data.train.dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)
