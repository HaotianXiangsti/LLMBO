import itertools
import math
import numpy as np
'''
In my code, each time involves calculating the delay and energy for a batch."

'''
class Edge_Server():


    def __init__(self, UDs, DNN, B, noisy, Fs, Ps):
        self.UDs = UDs  # system users
        self.DNN = DNN  # training DNN
        self.f_edge = 0         # current server frequency
        self.p_edge = 0         # current server transmit power
        self.bandwidth = B      # the uplink and downlink bandwidths of the dedicated spectral resource blocks allocated to each WD are fixed.
        self.AWGN = noisy
        self.xi = 0.5e-28         # the effective capacitance coefficient of edge server's computing chip
        self.epoch = 1          # training epoch
        self.Fs = Fs            # frequency throughout the entire training rounds
        self.Ps = Ps            # transmit power throughout the entire training rounds
        


    '''
    update edge server's f and p
    '''
    def info_update(self, S):

        self.S = S
        self.f_edge = self.Fs[self.epoch - 1]
        self.p_edge = self.Ps[self.epoch - 1]



        for idx, UD in enumerate(self.UDs):
            UD.update(int(self.S[idx]))


    '''
    EDC function
    '''
    def EDC(self, S):

        self.info_update(S)
        #print('Epoch:{}, f_edge:{}, p_edge:{}, split_points:{}'.format(self.epoch, self.f_edge, self.p_edge, self.S))

        self.epoch += 1
        return -(0.5 * self.delay() + 0.5 * self.energy())

    '''
    system delay: training delay, model downloading and uploading delay, smashed data and gradient delay
    '''
    def delay(self):
        D = 0

        for idx, UD in enumerate(self.UDs):
            d_total = self.d_train(UD) + self.d_trans(UD)
            # system's delay depends on the user with longest delay due to parallel
            if(d_total > D):
                D = d_total

        return D

    def d_train(self, UD):
        comp_ud = 0
        comp_edge = 0
        rho = UD.f * UD.cores / UD.cpi
        for i in range(self.DNN.N):
            if(i < UD.s):
                comp_ud += self.DNN.layer_comp_cost_info_forward[i]
                comp_ud += self.DNN.layer_comp_cost_info_backward[i]
            else:
                comp_edge += self.DNN.layer_comp_cost_info_forward[i]
                comp_edge += self.DNN.layer_comp_cost_info_backward[i]

        # print('UD:', UD.idx, 'train delay', comp_ud / rho + comp_edge / self.f_edge)

        return math.ceil(self.DNN.dataset_size / self.DNN.batch_size) * (comp_ud / rho + comp_edge / (self.f_edge / len(self.UDs)))

    def d_trans(self, UD):
        rate_device = self.bandwidth * math.log2(1 + (UD.p * UD.h) / self.AWGN)
        rate_edge = self.bandwidth * math.log2(1 + (self.p_edge * UD.h) / self.AWGN)
        d_data_trans = self.DNN.layer_output[UD.s] / rate_device + self.DNN.layer_gradient[UD.s] / rate_edge

        model_parameters = sum(self.DNN.paras[:UD.s])
        d_model_downloading = model_parameters / rate_edge
        d_model_uploading = model_parameters / rate_device
        d_para_trans = d_model_downloading + d_model_uploading

        return math.ceil(self.DNN.dataset_size / self.DNN.batch_size) * d_data_trans + d_para_trans


    def E_train(self, UD):
        comp_ud = 0
        comp_edge = 0
        for i in range(self.DNN.N):
            if (i < UD.s):
                comp_ud += self.DNN.layer_comp_cost_info_forward[i]
                comp_ud += self.DNN.layer_comp_cost_info_backward[i]
            else:
                comp_edge += self.DNN.layer_comp_cost_info_forward[i]
                comp_edge += self.DNN.layer_comp_cost_info_backward[i]

        E_UD = UD.k * UD.f ** 2 * comp_ud / UD.cores
        E_ES = self.xi * (self.f_edge / len(self.UDs)) ** 2 * comp_edge

        return math.ceil(self.DNN.dataset_size / self.DNN.batch_size) * (E_UD + E_ES)

    '''
    The energy consumption of smashed data including gradients and model parameters
    '''
    def E_trans(self, UD):

        # uplink transmission data rate
        rate_device = self.bandwidth * math.log2(1 + (UD.p * UD.h) / self.AWGN)
        # downlink transmission data rate
        rate_edge = self.bandwidth * math.log2(1 + (self.p_edge * UD.h) / self.AWGN)

        # energy consumption of smashed data and gradients
        e_data_trans = UD.p * self.DNN.layer_output[UD.s] / rate_device + self.p_edge * self.DNN.layer_gradient[UD.s] / rate_edge


        model_parameters = sum(self.DNN.paras[:UD.s])
        e_model_downloading = self.p_edge * model_parameters / rate_edge
        e_model_uploading = UD.p * model_parameters / rate_device
        e_para_trans = e_model_downloading + e_model_uploading

        return e_para_trans + math.ceil(self.DNN.dataset_size / self.DNN.batch_size) * (e_data_trans)


    def energy(self):

        energy_total = 0
        for idx, UD in enumerate(self.UDs):
            energy_total += self.E_train(UD) + self.E_trans(UD)


        return energy_total































