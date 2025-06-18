# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:50:20 2023

@author: xiao208
"""

#%%
import sys
import os
wd = 'E:\\Code\\simpl5pn_inhibition' # working directory
sys.path.insert(1, wd)

import numpy as np
import pickle
import sys
import time
import os
from func import comp_model
from func import parameters_three_com
from func import sequences
from func import post_analysis
P = parameters_three_com.init_params(wd)
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from func.l5_biophys import *
from func.param_fit import *
from scipy.stats import norm
from numpy.random import RandomState

def create_step_fun(dt, amp = np.asarray([0.0, 1.0]), T = 300, dur=100):
    t = np.arange(0, T+dt, dt)
    stp = np.arange(0, len(t), np.floor(dur/dt))
    stp = np.append(stp, len(t))
    stp = stp.astype(np.int32)
    g_temp = np.zeros(t.shape)
    for i in range(len(stp)-1):
        if i%2==0:
            g_temp[stp[i]:stp[i+1]] = amp[0]
        else:
            g_temp[stp[i]:stp[i+1]] = amp[1]
    return g_temp

def spon_spiking_inhibition(P, dt, T, S = np.asarray([0.0,1.0,2.0,3.0,4.0,5.0]), loc = np.asarray([0,2,3]), if_plot = 1, if_save = 1, base_dir = os.path.join(wd,'results'), save_dir = 'shunt'):

    cell = comp_model.CModel(P, verbool = False)
    v_init = -70.0

    rates_e = [np.zeros(P['N_e'])]
    rates_i =[ np.zeros(P['N_i'])]
    S_e = sequences.build_rate_seq(rates_e[0], 0, T)
    S_i = sequences.build_rate_seq(rates_i[0], 0, T)


    cell.P['dist'] = np.asarray([0.001,0.001,20.0,40.0])
    cell.P['g_na_p'] = 0
    cell.P['g_nad_p'] = 20-cell.P['g_na_p']
    cell.P['g_na_d'] = 0
    cell.P['g_nad_d'] = 20
    cell.P['N'] = np.asarray([np.inf, np.inf, np.inf, np.inf])
    #-------------- for subthreshold ---------------------
    # cell.P['kappa_d'] = 50
    # cell.P['kappa_p'] = 50
    # cell.P['g_ca_d'] = 0.1
    # cell.P['rho_p'] = 20
    # cell.P['rho_d'] = 50
    # cell.P['N'] = np.asarray([np.inf, np.inf, np.inf, np.inf])
    #-------------- for suprathreshold ---------------------
    cell = comp_model.CModel(P, verbool = False)

    v_init = -70.0
    v_all = []
    spike_time = []
    S1 = []
    loc1 = []
    mod = np.zeros((len(loc), len(S)))
    for (i,s) in enumerate(S):
        for (j,l) in enumerate(loc):
            g = create_step_fun(dt, np.asarray([0.0,s]), T, 200)
            t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj = [0.0], G_inj=[g], E = -77, Iinj_site = [0], Ginj_site = [l])
            S1.append(s)
            loc1.append(l)

            v = soln[0]
            t_spike = spike_times(dt, v)
            v_all.append(v)
            spike_time.append(t_spike)
            stim_idx = t[np.asarray([2000,6000,10000,14000]).astype(np.int32)]
            stimend_idx = t[np.asarray([4000,8000,12000]).astype(np.int32)]
            spike_stim = np.where(
            ((t_spike >= stim_idx[0]) & (t_spike < stimend_idx[0])) |
            ((t_spike >= stim_idx[1]) & (t_spike < stimend_idx[1])) |
            ((t_spike >= stim_idx[2]) & (t_spike < stimend_idx[2])) |
            (t_spike >= stim_idx[3]))[0]

            spike_nostim = np.where(
            (t_spike < stim_idx[0]) |
            ((t_spike >= stimend_idx[0]) & (t_spike < stim_idx[1])) |
            ((t_spike >= stimend_idx[1]) & (t_spike < stim_idx[2])) |
            ((t_spike >= stimend_idx[2]) & (t_spike < stim_idx[3])))[0]

            mod[j, i] = (len(spike_stim) / 0.3) / (len(spike_nostim) / 0.4)
            print('finishing with conductance %3f, location %d, modulation %3f \n'%(s, l, mod[j, i]))

            if if_plot:
                colors = np.asarray([[128,128,128],[61,139,191], [119,177,204], [6,50,99]])
                colors = colors/256
                plt.figure()
                ax = plt.subplot(211)
                for i in [0,2,3]:
                    ax.plot(t, soln[0][i], color = colors[i], linewidth=0.75)
                ax.set_xlim([0,500])
                comp = 2
                ax = plt.subplot(212)
                for i in [0,2,3]:
                    ax.plot(t, soln[1][7+i,comp,:].T, color = colors[i,:])
                ax.set_xlim([0,500])
                ax.set_ylim([0,1])
                plt.show()
    
    if if_save:
        datapath = os.path.join(base_dir, save_dir)
        today = date.today()
        now = datetime.now()
        current_date = today.strftime("%m%d%y")
        current_time = now.strftime("%H%M%S")
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        data_temp = {'v':v_all, 'S': S1,'T': T, 'dt': dt, 'g':g, 'loc': loc1, 'mod':mod}
        sio.savemat(os.path.join(datapath, 'shunt_%s_%s.mat'%(current_date, current_time)), data_temp)

def spon_spiking_inhibition_delayed_recovery(P, dt, T, S = np.asarray([0.0,1.0,2.0,3.0,4.0,5.0]), loc = np.asarray([0,2,3]), if_plot = 1, if_save = 1, base_dir = os.path.join(wd,'results'), save_dir = 'shunt_w_ionic'):

    cell = comp_model.CModel(P, verbool = False)
    v_init = -70.0

    rates_e = [np.zeros(P['N_e'])]
    rates_i =[ np.zeros(P['N_i'])]
    S_e = sequences.build_rate_seq(rates_e[0], 0, T)
    S_i = sequences.build_rate_seq(rates_i[0], 0, T)


    cell.P['dist'] = np.asarray([0.001,0.001,20.0,40.0])
    cell.P['g_na_p'] = 0
    cell.P['g_nad_p'] = 20-cell.P['g_na_p']
    cell.P['g_na_d'] = 0
    cell.P['g_nad_d'] = 20
    cell.P['N'] = np.asarray([np.inf, np.inf, np.inf, np.inf])
    
    #-------------- for subthreshold ---------------------
    # cell.P['kappa_d'] = 50
    # cell.P['kappa_p'] = 50
    # cell.P['g_ca_d'] = 0.1
    # cell.P['rho_p'] = 20
    # cell.P['rho_d'] = 50
    # cell.P['N'] = np.asarray([np.inf, np.inf, np.inf, np.inf])
    #-------------- for suprathreshold ---------------------
    cell = comp_model.CModel(P, verbool = False)

    v_init = -70.0
    v_all = []
    spike_time = []
    S1 = []
    loc1 = []
    mod = np.zeros((len(loc), len(S)))
    for (i,s) in enumerate(S):
        for (j,l) in enumerate(loc):
            if l==2:
                cell.P['dist'] = [0.001,0.001,create_step_fun(dt, np.asarray([10.0,10.0+30.0*(s/2.5)]), T, 200),40.0]
            elif l==3:
                cell.P['dist'] = [0.001,0.001,10.0,create_step_fun(dt, np.asarray([40.0,40.0+30.0*(s/2.5)]), T, 200)]
            elif l == 0:
                cell.P['g_na'] = create_step_fun(dt, np.asarray([3000.0,3000.0 - 1000.0*(s/2.5) ]), T, 200)
                cell.P['g_nad'] = create_step_fun(dt, np.asarray([0.0,1000.0*(s/2.5)  ]), T, 200) 
                cell.P['dist'] = np.array([10.0,0.001,10.0,40.0])
            g = create_step_fun(dt, np.asarray([0.0,s]), T, 200)
            t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj = [0.0], G_inj=[g], E = -77, Iinj_site = [0], Ginj_site = [l])
            S1.append(s)
            loc1.append(l)

            v = soln[0]
            t_spike = spike_times(dt, v)
            v_all.append(v)
            spike_time.append(t_spike)
            stim_idx = t[np.asarray([2000,6000,10000,14000]).astype(np.int32)]
            stimend_idx = t[np.asarray([4000,8000,12000]).astype(np.int32)]
            spike_stim = np.where(
            ((t_spike >= stim_idx[0]) & (t_spike < stimend_idx[0])) |
            ((t_spike >= stim_idx[1]) & (t_spike < stimend_idx[1])) |
            ((t_spike >= stim_idx[2]) & (t_spike < stimend_idx[2])) |
            (t_spike >= stim_idx[3]))[0]

            spike_nostim = np.where(
            (t_spike < stim_idx[0]) |
            ((t_spike >= stimend_idx[0]) & (t_spike < stim_idx[1])) |
            ((t_spike >= stimend_idx[1]) & (t_spike < stim_idx[2])) |
            ((t_spike >= stimend_idx[2]) & (t_spike < stim_idx[3])))[0]

            mod[j, i] = (len(spike_stim) / 0.3) / (len(spike_nostim) / 0.4)
            print('finishing with conductance %3f, location %d, modulation %3f \n'%(s, l, mod[j, i]))

            if if_plot:
                colors = np.asarray([[128,128,128],[61,139,191], [119,177,204], [6,50,99]])
                colors = colors/256
                plt.figure()
                ax = plt.subplot(211)
                for i in [0,2,3]:
                    ax.plot(t, soln[0][i], color = colors[i], linewidth=0.75)
                ax.set_xlim([0,500])
                comp = 2
                ax = plt.subplot(212)
                for i in [0,2,3]:
                    ax.plot(t, soln[1][7+i,comp,:].T, color = colors[i,:])
                ax.set_xlim([0,500])
                ax.set_ylim([0,1])
                plt.show()
    
    if if_save:
        if not os.path.exists(os.path.join(base_dir, save_dir)):
            # Create the directory
            os.makedirs(os.path.join(base_dir, save_dir))
        datapath = os.path.join(base_dir, save_dir)
        today = date.today()
        now = datetime.now()
        current_date = today.strftime("%m%d%y")
        current_time = now.strftime("%H%M%S")
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        data_temp = {'v':v_all, 'S': S1,'T': T, 'dt': dt, 'g':g, 'loc': loc1, 'mod':mod}
        sio.savemat(os.path.join(datapath, 'shunt_%s_%s.mat'%(current_date, current_time)), data_temp)

def spon_spiking_delayed_recovery(P, dt, T, S = np.asarray([0.0,1.0,2.0,3.0,4.0,5.0]), loc = np.asarray([0,2,3]), if_plot = 1, if_save = 1, base_dir = os.path.join(wd,'results'), save_dir = 'ionic'):

    cell = comp_model.CModel(P, verbool = False)
    v_init = -70.0

    rates_e = [np.zeros(P['N_e'])]
    rates_i =[ np.zeros(P['N_i'])]
    S_e = sequences.build_rate_seq(rates_e[0], 0, T)
    S_i = sequences.build_rate_seq(rates_i[0], 0, T)


    cell.P['dist'] = np.asarray([0.001,0.001,20.0,40.0])
    cell.P['g_na_p'] = 0
    cell.P['g_nad_p'] = 20-cell.P['g_na_p']
    cell.P['g_na_d'] = 0
    cell.P['g_nad_d'] = 20
    cell.P['N'] = np.asarray([np.inf, np.inf, np.inf, np.inf])
    
    #-------------- for subthreshold ---------------------
    # cell.P['kappa_d'] = 50
    # cell.P['kappa_p'] = 50
    # cell.P['g_ca_d'] = 0.1
    # cell.P['rho_p'] = 20
    # cell.P['rho_d'] = 50
    # cell.P['N'] = np.asarray([np.inf, np.inf, np.inf, np.inf])
    #-------------- for suprathreshold ---------------------
    cell = comp_model.CModel(P, verbool = False)

    v_init = -70.0
    v_all = []
    spike_time = []
    S1 = []
    loc1 = []
    mod = np.zeros((len(loc), len(S)))
    for (i,s) in enumerate(S):
        for (j,l) in enumerate(loc):
            if l==2:
                cell.P['dist'] = [0.001,0.001,create_step_fun(dt, np.asarray([10.0,10.0+30.0*(s/2.5)]), T, 200),40.0]
            elif l==3:
                cell.P['dist'] = [0.001,0.001,10.0,create_step_fun(dt, np.asarray([40.0,40.0+30.0*(s/2.5)]), T, 200)]
            elif l == 0:
                cell.P['g_na'] = create_step_fun(dt, np.asarray([3000.0,3000.0 - 1000.0*(s/2.5) ]), T, 200)
                cell.P['g_nad'] = create_step_fun(dt, np.asarray([0.0,1000.0*(s/2.5)  ]), T, 200) 
                cell.P['dist'] = np.array([10.0,0.001,10.0,40.0])
            g = create_step_fun(dt, np.asarray([0.0,s]), T, 200)
            g = np.zeros(g.shape)
            t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj = [0.0], G_inj=[g], E = -77, Iinj_site = [0], Ginj_site = [l])
            S1.append(s)
            loc1.append(l)

            v = soln[0]
            t_spike = spike_times(dt, v)
            v_all.append(v)
            spike_time.append(t_spike)
            stim_idx = t[np.asarray([2000,6000,10000,14000]).astype(np.int32)]
            stimend_idx = t[np.asarray([4000,8000,12000]).astype(np.int32)]
            spike_stim = np.where(
            ((t_spike >= stim_idx[0]) & (t_spike < stimend_idx[0])) |
            ((t_spike >= stim_idx[1]) & (t_spike < stimend_idx[1])) |
            ((t_spike >= stim_idx[2]) & (t_spike < stimend_idx[2])) |
            (t_spike >= stim_idx[3]))[0]

            spike_nostim = np.where(
            (t_spike < stim_idx[0]) |
            ((t_spike >= stimend_idx[0]) & (t_spike < stim_idx[1])) |
            ((t_spike >= stimend_idx[1]) & (t_spike < stim_idx[2])) |
            ((t_spike >= stimend_idx[2]) & (t_spike < stim_idx[3])))[0]

            mod[j, i] = (len(spike_stim) / 0.3) / (len(spike_nostim) / 0.4)
            print('finishing with conductance %3f, location %d, modulation %3f \n'%(s, l, mod[j, i]))

            if if_plot:
                colors = np.asarray([[128,128,128],[61,139,191], [119,177,204], [6,50,99]])
                colors = colors/256
                plt.figure()
                ax = plt.subplot(211)
                for i in [0,2,3]:
                    ax.plot(t, soln[0][i], color = colors[i], linewidth=0.75)
                ax.set_xlim([0,500])
                comp = 2
                ax = plt.subplot(212)
                for i in [0,2,3]:
                    ax.plot(t, soln[1][7+i,comp,:].T, color = colors[i,:])
                ax.set_xlim([0,500])
                ax.set_ylim([0,1])
                plt.show()
    
    if if_save:
        if not os.path.exists(os.path.join(base_dir, save_dir)):
            # Create the directory
            os.makedirs(os.path.join(base_dir, save_dir))
        datapath = os.path.join(base_dir, save_dir)
        today = date.today()
        now = datetime.now()
        current_date = today.strftime("%m%d%y")
        current_time = now.strftime("%H%M%S")
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        data_temp = {'v':v_all, 'S': S1,'T': T, 'dt': dt, 'g':g, 'loc': loc1, 'mod':mod}
        sio.savemat(os.path.join(datapath, 'shunt_%s_%s.mat'%(current_date, current_time)), data_temp)

dt = 0.1
T = 1400

S = np.arange(0.0,2.75,0.25)
spon_spiking_delayed_recovery(P, dt, T, S = S, if_plot = 0, base_dir = os.path.join(wd,'results'))



