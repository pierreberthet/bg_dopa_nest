import os
import numpy as np
import json
import ParameterContainer

class BasalGangliaParameters(ParameterContainer.ParameterContainer):
    def __init__(self, params_fn=None):
        super(BasalGangliaParameters, self).__init__(params_fn)
        self.params = {}
	if params_fn == None:
            self.set_default_params()


    def set_default_params(self):

        self.set_bg_params()


    def set_bg_params(self):
        """
        Parameters for Basal Ganglia        
        """

        self.params['n_actions'] = 11
        self.params['n_states'] = 100


        ## STR
        self.params['model_exc_neuron'] = 'iaf_cond_alpha_bias'
        self.params['model_inh_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_msn_d1'] = 30
        self.params['num_msn_d2'] = 30
        self.params['param_msn_d1'] = {'fmax':20.0, 'tau_j': 10.,'tau_e': 100.,'tau_p':100000., 'epsilon': 0.01, 't_ref': 2.0, 'gain': 0.0}
        self.params['param_msn_d2'] = {'fmax':20.0, 'tau_j': 10.,'tau_e': 100.,'tau_p':100000., 'epsilon': 0.01, 't_ref': 2.0, 'gain': 0.0}

        
        ## Output GPi/SNr
        self.params['model_bg_output_neuron'] = 'iaf_cond_alpha'
        self.params['num_actions_output'] = 1
        self.params['param_bg_output'] = {} # to adapt parms to aif_cond_alpha neuron model
        
        self.params['str_to_output_exc_w'] = 10.
        self.params['str_to_output_inh_w'] = -10.
        self.params['str_to_output_exc_delay'] = 1. 
        self.params['str_to_output_inh_delay'] = 1.
        
        ## RP and REWARD
        self.params['model_rp_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_rp_neurons'] = 15
        self.params['param_rp_neuron'] = {'fmax':20., 'tau_j': 10.,'tau_e': 100.,'tau_p':100000., 'epsilon': 0.01, 't_ref': 2., 'gain': 0.}

        self.params['model_rew_neuron'] = 'iaf_cond_alpha'
        self.params['num_rew_neurons'] = 20
        self.params['param_rew_neuron'] = {} # to adapt parms to aif_cond_alpha neuron model
        self.params['model_poisson_rew'] = 'poisson_generator'
        self.params['num_poisson_rew'] = 20
        self.params['weight_poisson_rew'] = 10.
        self.params['delay_poisson_rew'] = 1.
        self.params['param_poisson_rew'] = {}# to adapt parms to aif_cond_alpha neuron model

        self.params['weight_rp_rew'] = -5. #inhibition of the dopaminergic neurons in rew by the current reward prediction from rp[current state, selected action]
        self.params['delay_rp_rew'] = 1.


        #Connections Actions and States to RP
        self.epsilon = 0.01
        self.tau_i = 10.
        self.tau_j = 10.
        self.tau_e = 100.
        self.tau_p = 100000.

        self.params['actions_rp'] = 'bcpnn_synapse'
        self.params['param_actions_rp'] = {'gain': 0.70, 'K':1.0,'fmax': 20.0,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        self.params['states_rp'] = 'bcpnn_synapse'
        self.params['param_states_rp'] = {'gain': 0.70, 'K':1.0,'fmax': 20.0,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}

            self.params['bcpnn'] = 'bcpnn_synapse'
        self.params['param_bcpnn'] =  {'gain': 0.70, 'K':1.0,'fmax': 20.0,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}

        #Connections States Actions
        self.params['synapse_d1_MT_BG'] = 'bcpnn_synapse'
        self.params['params_synapse_d1_MT_BG'] = {'gain': 0.70, 'K':1.0,'fmax': 20.0,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        self.params['synapse_d2_MT_BG'] = 'bcpnn_synapse'
        self.params['params_synapse_d2_MT_BG'] = {'gain': 0.70, 'K':1.0,'fmax': 20.0,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}

        #Connections REW to RP, STRD1 and STRD2
        self.params['weight_rew_strD1'] = 10.
        self.params['weight_rew_strD2'] = 10.
        self.params['delay_rew_strD1'] = 1.
        self.params['delay_rew_strD2'] = 1.

        self.params['weight_rew_rp'] = 10.
        self.params['delay_rew_rp'] = 1.

        #Spike detectors params 	
        self.params['spike_detector_output_action'] = {"withgid":True, "withtime":True} 

        #Supervised Learning
        self.params['supervised_on'] = True

        self.params['num_neuron_poisson_supervisor'] = 30
        self.params['num_neuron_poisson_input_BG'] = 50
        self.params['active_supervisor_rate'] = 25.
        self.params['inactive_supervisor_rate'] = 0.
        self.params['active_poisson_input_rate'] = 20.
        self.params['inactive_poisson_input_rate'] = 2.
        
        self.params['param_poisson_pop_input_BG'] = {}
        self.params['param_poisson_supervisor'] = {}

        self.params['weight_supervisor_strd1'] = 10.
        self.params['weight_supervisor_strd2'] = 10.
        self.params['delay_supervisor_strd1'] = 1.
        self.params['delay_supervisor_strd2'] = 1.

        self.params['weight_poisson_input'] = 10.
        self.params['delay_poisson_input'] = 1.

        self.params['are_MT_BG_connected'] = False
        
        self.params['num_neuron_states'] = 20
        self.params['param_states_pop'] = {} 
    """
        self.params[' '] = 
        self.params[' '] = 
        self.params[' '] = 
        self.params[' '] = 
        self.params[' '] = 
        self.params[' '] = 
        self.params[' '] = 
    """
