#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:28:26 2026

@author: aeferretti
"""

import os

import matplotlib.pyplot as plt
import np as np

import jax
import jax.numpy as jnp
import optax

import forward

from time import time as time

#Utility functions 
def keep_iterating(loss_hist, max_hist=1000, w=10, thresh=1e-3):
    if len(loss_hist) > max_hist:
        return False
    elif len(loss_hist) > w:
        return (np.abs(loss_hist[-1] - loss_hist[-w]) >= thresh)
    return True

def overlap_penalty(recons, axes=(0,1)):
    im1, im2 = recons[axes[0]], recons[axes[0]]
    return jnp.mean(im1 * im2)

def clip_constraint(recons):
    return jnp.clip(recons, 0.0, 2.0)

def mask_constraint(recons, mask):
    return jnp.where(mask, recons, 0.0)

def plot_recons(opt_recons, gt_arr, loss_history):
    fig, ax = plt.subplots(1, 5, figsize=[16, 3], dpi=100, layout='constrained')

    # recon panels
    for i in range(2):
        ax[i].set_title(f'mat {i+1} recon')
        m = ax[i].imshow(opt_recons[i], aspect='equal', cmap='viridis')
        fig.colorbar(m, ax=ax[i])

    # recon - ground truth panels
    for i in range(2):
        err = opt_recons[i] - gt_arr[i]
        vmax = np.max(np.abs(np.asarray(err)))

        ax[i + 2].set_title(f'mat {i+1} error')
        m = ax[i + 2].imshow(err, aspect='equal', cmap='bwr', vmin=-vmax, vmax=vmax)
        fig.colorbar(m, ax=ax[i + 2])

    # loss panel
    ax[4].set_title('loss')
    ax[4].plot(loss_history, '.-')
    ax[4].set_xlabel('iteration')
    ax[4].set_yscale('log')

    plt.show()


class iterative_recon():
    def __init__(self):
        self.loss_function = None
        self.optimizer = None
        self.recon_model = None
        self.constraints = None
        self.loss_history = []
        self.iteration_no = 0
        self.aux = None
        return 
        
    def set_data(self,de_data_counts):
        self.de_data_counts = de_data_counts
    
    def set_recon_model(self,model,opt_pars):
        if model == 'PA':
            recon_model = forward.ProjApproxPBI_1D(**opt_pars)
            recon_fwd = jax.jit(
                lambda material_maps: recon_model.apply({}, material_maps)
            )
            self.recon_model = recon_fwd
        elif model == 'MA':
            recon_model = forward.MultiSlicePBI_1D(**opt_pars)
            recon_fwd = jax.jit(
                lambda material_maps: recon_model.apply({}, material_maps)
            )
            self.recon_model = recon_fwd
        else:
            print('Please specify model PA or MA')
        return

        
    
    def config_loss_fn(self,reg_list,reg_weights):
        #reg_list should be a iterable of regularizing functions that take the recon
        #Volumes and return a scalar penalty
        #reg_weights must be a iterable of the same length
        #Modify this so no regularizer can be passed 
        
        if self.recon_model != None:
            
            reg_fun = lambda recon_vol: [reg_list[i](recon_vol)*reg_weights[i] for i in range(len(reg_list))]
            
            def loss_fn_class(recons, data):
                y = self.recon_model(recons)
                fidelity = jnp.mean((y - data)**2)  
                reg = reg_fun(recons)
                loss = fidelity + jnp.sum(jnp.array(reg))
                aux = {
                    'fidelity': fidelity,
                    'reg': reg,
                    'loss': loss,
                }
                return loss, aux
            self.loss_function = loss_fn_class
            return
            
    
    def config_optimizer(self,lrate,inital_values):
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lrate),
        )
        
        self.optimizer = optimizer
        self.opt_recons = inital_values
        self.opt_state = optimizer.init(inital_values)
        return 
    
    
    def create_train_step(self,const_funs=None,const_input=None):
        #Const_fun is a list of constraint functions, each of which can accept up to 2 arguments
        #The first argument must be the recon volume 
        #The second can be anything. If more than one constant input is needed, 
        #use a list or other iterable, and unpack it inside the constraint function 
        #Constraints applied in the order of const_funs,
        if const_funs is None:
            @jax.jit
            def train_step(recons, opt_state, data):
                (loss, aux), grads = jax.value_and_grad(self.loss_function, has_aux=True)(recons, data)
                
                updates, opt_state = self.optimizer.update(grads, self.opt_state, recons)
                recons = optax.apply_updates(recons, updates)
                
                return recons, opt_state, loss, aux
            self.train_step_fun = train_step
            
        else:
            con_fun_list = []
            for i in range(len(const_funs)):     
                if const_input[i] is None:
                    con_fun = lambda recons: const_funs[i]
                    con_fun_list.append(con_fun)
                else:
                    con_fun = lambda recons: const_funs[i](recons,const_input)
                    con_fun_list.append(con_fun)
                    
            @jax.jit
            def train_step(recons, opt_state, data):
                (loss, aux), grads = jax.value_and_grad(self.loss_function, has_aux=True)(recons, data)
                
                updates, opt_state = self.optimizer.update(grads, self.opt_state, recons)
                recons = optax.apply_updates(recons, updates)
                
                #Apply constraints 
                for i in range(len(con_fun_list)):
                    recons = con_fun_list(recons)

                return recons, opt_state, loss, aux
            self.train_step_fun = train_step
        return


    
    def do_iter(self):
        opt_recons, opt_state, loss, aux = self.train_step_fun(self.opt_recons, self.opt_state, self.de_data_counts)  # make sure data is correct!!
        self.loss_history.append(float(aux['loss']))
        self.opt_recons = opt_recons
        self.opt_state = opt_state
        self.aux = aux 
        self.iteration_no = self.iteration_no + 1
        
    def do_recon(self,report_status_itr=20):
        while keep_iterating(self.loss_history, max_hist=500, w=5, thresh=1e-3):    
            self.do_iter()
            if (self.iteration_no % report_status_itr == 0):
                print(
                    f'iter {self.iteration_no}: '
                    f'loss = {float(self.aux["loss"]):.4e}, '
                    f'fidelity = {float(self.aux["fidelity"]):.4e}, '
                    f'reg = {float(self.aux["reg"]):.4e}'
                )
            print(self.iteration_no)
            
    def plot_recons(self,gt_arr):
        plot_recons(self.opt_recons, gt_arr, self.loss_history)