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
from func import parameters_two_com
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



cell = comp_model.CModel(P, verbool = False)
v_init = -70.0
dt = 0.1
T = 1500

# dt = 0.05
# T = 700

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
cell.P['kappa_d'] = 50
cell.P['kappa_p'] = 50
cell.P['g_ca_d'] = 0.1
cell.P['rho_p'] = 20
cell.P['rho_d'] = 50
#-------------- for subthreshold ---------------------

# g = create_step_fun(dt, np.asarray([0.0,5.0]), T, 500)
i_s = create_step_fun(dt, np.asarray([-7.5,-7.5]), T, 500)
i_d = create_step_fun(dt, np.asarray([0,-7.5]), T, 500)
g = create_step_fun(dt, np.asarray([0.0,5.0]), T, 100)
t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj = [-2.0,0], G_inj=[g], E = -77, Iinj_site = [0,3], Ginj_site = [2])

colors = np.asarray([[128,128,128],[61,139,191], [119,177,204], [6,50,99]])
colors = colors/256
plt.figure()
v = soln[0]
t_spike = spike_times(dt, v)
t_dspike = d_spike_times(dt,v, t_spike)
for i in [0,2,3]:
    plt.plot(t, soln[0][i], color = colors[i], linewidth=0.75)
# plt.scatter(t_spike, v[0, np.floor(t_spike/dt).astype(np.int32)])
# plt.scatter(t_dspike.T[0], v[2, np.floor(t_dspike.T[0]/dt).astype(np.int32)])
# plt.scatter(t_dspike.T[1], v[3, np.floor(t_dspike.T[1]/dt).astype(np.int32)])
plt.xlim([0,T])
plt.xlim([250,1250])
plt.ylim([-100,-70])
plt.show()





# %%
