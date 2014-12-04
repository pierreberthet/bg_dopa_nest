import os
import numpy as np
import json
import ParameterContainer
from copy import deepcopy as dc
import pprint

class global_parameters(ParameterContainer.ParameterContainer):
#class global_parameters(object):
    """
    The parameter class storing the simulation parameters 
    is derived from the ParameterContainer class.

    Parameters used (mostly) by different classes should be seperated into
    different functions.
    Common parameters are set in the set_default_params function
    """

    def __init__(self, args=None,  params_fn=None):#, output_dir=None):
        """
        Keyword arguments:
        params_fn -- string, if None: set_filenames and set_default_params will be called
        """
        

        if params_fn == None:
            self.params = {}
            self.set_default_params()
            if args!=None:
                self.set_parall_param(args)
            self.set_dependables()
            self.set_bg_params()
        else:
            self.load_params_from_file(params_fn)
        super(global_parameters, self).__init__() # call the constructor of the super/mother class


    def set_parall_param(self, args):
        print 'block length:', self.params['block_len']
        self.params['block_len'] = int(args[0])
        print 'block length:', self.params['block_len']

        self.params['rank_multi'] = int(args[1])
        self.params['multi_n'] = int(args[2])


    def set_dependables(self):
        self.params['t_iteration'] = self.params['t_selection'] + self.params['t_efference'] + self.params['t_rest'] + self.params['t_reward']  # [ms] stimulus integration time, 
        self.params['t_sim'] = self.params['t_init'] + self.params['t_iteration'] * self.params['block_len'] * self.params['n_blocks']          # [ms] total simulation time
        self.params['n_iterations'] = self.params['block_len'] * self.params['n_blocks']  #int(round(2*self.params['t_sim'] / self.params['t_iteration']))
        self.params['n_recordings'] = self.params['t_sim'] / self.params['resolution']
        print 'N_recordings = ', self.params['n_recordings']


    def set_default_params(self):
        """
        Here all the simulation parameters NOT being filenames are set.
        """

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_selection'] = 500.
        self.params['t_efference'] = 250.
        self.params['t_reward'] = 250. #70.
        self.params['t_rest'] = 500. 
        self.params['t_delay'] = 0.
        self.params['t_init'] = 2500.
        # time % resolution has to be 0
        self.params['resolution'] = 250.
        # after this time the input stimulus will be transformed
        self.params['block_len'] = 10
        self.params['n_blocks'] = 3        
        self.params['dt'] = .1                      # [ms] /default .1
        self.params['dt_input_mpn'] = 0.1           # [ms] time step for the inhomogenous Poisson process for input spike train generation

        self.params['record_spikes'] = True
        self.params['record_voltages'] = False
        self.params['light_record'] = True
        self.params['softmax'] =True 

        self.params['threshold']= 0.05

        # ##############################
        # BASAL GANGLIA PARAMETERS
        # ##############################

    def set_bg_params(self):
        """
        Parameters for Basal Ganglia        
        """
        self.params['n_states'] = 3
        self.params['n_actions'] = 3

        self.params['random_divconnect_poisson'] = 1.           # set to 1. if num poisson == 1, otherwise lower value? (was 0.75)
        self.params['random_connect_voltmeter'] = 0.01
        
        self.params['epsilon'] = 0.0001
        self.params['tau_i'] = 5.
        self.params['tau_j'] = 6.
        self.params['tau_e'] = 40.
        self.params['tau_p'] = 200. #1000.
        self.params['gain'] = 2.5 ##1.9   #3.
        self.params['gain_d1'] = 3. ##1.9   #3.
        self.params['gain_d2'] = 1.5 ##1.9   #3.
        self.params['gain_rp'] = -3.
        self.params['gain_neuron'] = 1.       #gain for the neuron model has different impact (amplifies current injected) than gain in synapse model (amplifies weights)
        self.params['K'] = 0.
        self.params['fmax'] = 40.  #70
        self.params['rp_fmax'] = 40.   #self.params['fmax']
        self.params['Vth'] = -50.
        self.params['Cm'] = 250.
        self.params['Vreset'] = -75.
        self.params['gL'] = 16.6667
        self.params['temperature'] = 1.

        self.params['Cm_std'] = 25.
        self.params['Vth_std'] = 2.
        self.params['Vreset_std'] = 2.
        self.params['rpe'] = 0.01

        self.params['trigger']= False
        self.params['block_trigger'] = 10
        self.params['value_trigg'] = 30. #percentage of dopamine neurons silenced by the disease
        self.params['new_value_2'] = 2450.
       # self.params['active_poisson_rew_rate'] = 2700.
       # self.params['baseline_poisson_rew_rate'] = 2500.
       # self.params['inactive_poisson_rew_rate'] = 2300.
        # ========================
        # Striatum MSN 
        # ========================

        self.params['model_state_neuron'] = 'iaf_cond_alpha'

        self.params['model_exc_neuron'] = 'iaf_cond_alpha_bias'
        self.params['model_inh_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_msn_d1'] = 30
        self.params['num_msn_d2'] = 30
        self.params['param_msn_d1'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm'], 'kappa': self.params['K'] ,'fmax':self.params['fmax'],'V_reset': self.params['Vreset'], 
                'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p':self.params['tau_p'], 'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gain']}
        self.params['param_msn_d2'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm'], 'kappa': self.params['K'] ,'fmax':self.params['fmax'],'V_reset': self.params['Vreset'], 
                'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p':self.params['tau_p'], 'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gain']}

        
        # ========================
        # GPi / SNr 
        # ========================
        self.params['model_bg_output_neuron'] = 'iaf_cond_alpha'
        self.params['num_actions_output'] = 10
        self.params['param_bg_output'] = {'V_reset': self.params['Vreset']} # to adapt parms to aif_cond_alpha neuron model
        
        self.params['str_to_output_exc_w'] = 1.         ### D1
        self.params['str_to_output_inh_w'] = -1.        ### D2
        self.params['str_to_output_exc_delay'] = 1. 
        self.params['str_to_output_inh_delay'] = 1.
        

        # ========================
        # RP and REWARD 
        # ========================
        self.params['model_rp_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_rp_neurons'] = 15
        self.params['param_rp_neuron'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm'], 'kappa': self.params['K'] ,'fmax':self.params['fmax'], 'V_reset': self.params['Vreset'],
                'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p':self.params['tau_p'], 'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gain']}

        
        self.params['num_rew_neurons'] = 200  # * 12?
        self.params['model_rew_neuron'] = 'iaf_cond_alpha_bias'
        self.params['param_rew_neuron'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm']} 
                                            # to adapt parms to aif_cond_alpha neuron model

        self.params['weight_rp_rew'] = -0. # inhibition of the dopaminergic neurons in rew by the 
                                           # current reward prediction from rp[current state, selected action]
        self.params['delay_rp_rew'] = 1.
        self.params['vt_params'] = {}


        # ========================
        # CONNECTIONS 
        # ========================
        #Connections Actions and States to RP
        # parameters for the standard bcpnn connections
        self.params['p_i'] = np.random.random()/33.
        self.params['p_j'] = np.random.random()/33.
        self.params['p_ij']= self.params['p_i'] * self.params['p_j']
        self.params['p_ij']= (np.random.random()/33.)*(np.random.random()/33.)
        print 'InitialP Pi', self.params['p_i'], 'pj',self.params['p_j'] , 'Pij', self.params['p_ij'] 
       # self.params['p_i'] = self.params['t_selection'] / ( self.params['n_states']*self.params['t_iteration']  ) 
       # self.params['p_j'] = ( self.params['t_reward'] + self.params['t_efference'] ) / ( self.params['n_actions']*self.params['t_iteration']  )
       # self.params['p_ij']= self.params['p_i'] * self.params['p_j']

        self.params['p_i_std']= self.params['p_i']/10.
        self.params['p_j_std']= self.params['p_j']/10.
        self.params['p_ij_std']= self.params['p_ij']/10.
        
        # ========================
        #  
        # ========================
        # parameters for RP pathway. Initial expectation should be 0.5
        self.params['p_ir'] = 0.01
        self.params['p_jr'] = 0.01
        self.params['p_ijr']= 0.0001

        self.params['actions_rp'] = 'bcpnn_dopamine_synapse'
        self.params['param_actions_rp'] = {'p_i': self.params['p_ir'], 'p_j': self.params['p_jr'], 'p_ij': self.params['p_ijr'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']}
        self.params['states_rp'] = 'bcpnn_dopamine_synapse'
        self.params['param_states_rp'] = {'p_i': self.params['p_ir'], 'p_j': self.params['p_jr'], 'p_ij': self.params['p_ijr'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':self.params['t_selection']+self.params['t_efference'],
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']}

        self.params['bcpnn'] = 'bcpnn_synapse'
        self.params['param_bcpnn'] = {'p_i': self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'],
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']} 
        self.params['lateral_synapse_d1'] = 'bcpnn_inhib_d1'
        self.params['lateral_synapse_d2'] = 'bcpnn_inhib_d2'
        self.params['params_lateral_synapse_d1'] = {}
        self.params['params_lateral_synapse_d2'] = {}

        self.params['inhib_lateral_weights_d1'] = -2. #-4.
        self.params['inhib_lateral_weights_d2'] = -3. #-4.
        self.params['inhib_lateral_weights_d2_d1'] = -1. #-4.
        self.params['inhib_lateral_delay_d1'] = 1.
        self.params['inhib_lateral_delay_d2'] = 1.
        # during learning gain == 0. K = 1.0 : --> 'offline' learning
        # after learning: gain == 1. K = .0

        # ========================
        # Dopa BCPNN parameters 
        # ========================
        #Connections States Actions
        
        self.params['synapse_d1'] = 'bcpnn_dopamine_synapse_d1'
        self.params['params_synapse_d1'] = {'p_i': self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']} 
        self.params['synapse_d2'] = 'bcpnn_dopamine_synapse_d2'
        self.params['params_synapse_d2'] = {'p_i': self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']} 

        # ========================
        # Dopa BCPNN parameters 
        # ========================
        #Connections REW to RP, STRD1 and STRD2
        self.params['weight_rew_strD1'] = 4.
        self.params['weight_rew_strD2'] = 4.
        self.params['delay_rew_strD1'] = 1.
        self.params['delay_rew_strD2'] = 1.

        self.params['weight_rew_rp'] = 4.
        self.params['delay_rew_rp'] = 1.

        self.params['w_rew_vtdopa'] = 1.
        self.params['delay_rew_vtdopa'] = 1.


        # ========================
        # Spike DETECTORS parameters
        # ========================

        self.params['spike_detector_action'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_d1'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_d2'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_states'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_efference'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_rew'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_rp'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_brainstem'] = {"withgid":True, "withtime":True}

        self.params['spike_detector_test_rp'] = {"withgid":True, "withtime":True}
 
        # ========================
        # POISSON INPUTS RATES 
        # ========================

	    #Reinforcement Learning


        self.params['num_neuron_poisson_efference'] = 1
        self.params['num_neuron_poisson_input_BG'] =1 
        self.params['active_full_efference_rate'] = 1200.  #1800. #2000.#600.  #3000.
        self.params['inactive_efference_rate'] = 1.
        self.params['active_poisson_input_rate'] = 1550.   #1900.  #2500.
        self.params['inactive_poisson_input_rate'] = 1200. #1400.
        self.params['supervisor_off'] = 0.

        self.params['active_poisson_rew_rate'] = 2700.
        self.params['baseline_poisson_rew_rate'] = 2500.
        self.params['inactive_poisson_rew_rate'] = 2300.  #2300.
        
        #Initialisation####################
        self.params['initial_poisson_input_rate'] = 1.  #2500.
        self.params['initial_noise_striatum_rate'] = 5500.  #2500.

        self.params['param_poisson_pop_input_BG'] = {}
        self.params['param_poisson_efference'] = {}
        
        self.params['model_poisson_rew'] = 'poisson_generator'
        self.params['num_poisson_rew'] = 1
        self.params['weight_poisson_rew'] = 4.
        self.params['delay_poisson_rew'] = 1.
        self.params['param_poisson_rew'] = {}# to adapt parms to aif_cond_alpha neuron model

        self.params['weight_efference_strd1_exc'] = 2.      ## 4.
        self.params['weight_efference_strd1_inh'] = -1.     ## -2.
        self.params['weight_efference_strd2_exc'] = 2.
        self.params['weight_efference_strd2_inh'] = -1.
        self.params['delay_efference_strd1_exc'] = 1.
        self.params['delay_efference_strd1_inh'] = 1.
        self.params['delay_efference_strd2_exc'] = 1.
        self.params['delay_efference_strd2_inh'] = 1.

        self.params['weight_poisson_input'] = 4.
        self.params['delay_poisson_input'] = 1.
        
        self.params['num_neuron_states'] = 30
        self.params['param_states_pop'] = {} 

        self.params['weight_states_rp'] = 5.
        self.params['delay_states_rp'] =   self.params['t_efference']
        self.params['weight_efference_rp'] = 5.
        self.params['delay_efference_rp'] =  self.params['t_efference']
       # self.params['weight_actions_rp'] = 5.
       # self.params['delay_actions_rp'] =  self.params['t_efference']

        # ========================
        # Dopa BCPNN parameters 
        # ========================
        

        self.params['gain_dopa'] = 7.  #5.   #4.
        self.params['dopa_b'] = - .0697   #####.069 #-.085  #-1.4      #-0.13   # - (baseline rate dopa (= pop size * rate) )/ 1000
        self.params['weight'] = 5.

        self.params['dopa_bcpnn'] = 'bcpnn_dopamine_synapse'
        self.params['params_dopa_bcpnn'] ={
            'bias':0.0,   #ANN interpretation. Only calculated here to demonstrate match to rule. 
                          # Will be eliminated in future versions, where bias will be calculated postsynaptically       
            'b':self.params['dopa_b'],
            'delay':1.,
            'dopamine_modulated':True,
            'complementary':False, #set to use the complementary traces or not
            'e_i':0.01,
            'e_j':0.01,
            'e_j_c':.99,
            'e_ij':0.001,
            'e_ij_c':0.3,
            'epsilon':0.001, #lowest possible probability of spiking, e.g. lowest assumed firing rate
            'fmax':self.params['fmax'],    #Frequency assumed as maximum firing, for match with abstract rule
            'gain':self.params['gain'],    #Coefficient to scale weight as conductance, can be zero-ed out
            'gain_dopa':self.params['gain_dopa'], 
            'k_pow': 3.,
            'K':0.,         #Print-now signal // Neuromodulation. Turn off learning, K = 0
            'sigmoid':0.,
            'sigmoid_mean':0.,
            'sigmoid_slope':1.,
            'n': .07, #17,
            'p_i': self.params['p_i'],   #0.01,
            'p_j':  self.params['p_j'],  #0.01,
            'p_ij':  self.params['p_ij'], #0.0001,
            'reverse':1., #1. 
            'tau_i':self.params['tau_i'],     #Primary trace presynaptic time constant
            'tau_j':self.params['tau_j'],      #Primary trace postsynaptic time constant
            'tau_e':self.params['tau_e'],      #Secondary trace time constant
            'tau_p':self.params['tau_p'],     #Tertiarty trace time constant
            'tau_n':300.,    #default 100
            #'type_id':'bcpnn_dopamine_synapse',
            'weight':self.params['weight'],
            'z_i':0.3,
            'z_j':0.3,
            'alpha': self.params['n_actions'] + 0. ,
            'value':1.,}

        self.params['params_dopa_bcpnn_d1'] = dc(self.params['params_dopa_bcpnn'])
        self.params['params_dopa_bcpnn_d1']['gain'] = self.params['gain_d1']

        self.params['params_dopa_bcpnn_d2'] = dc(self.params['params_dopa_bcpnn'])
        self.params['params_dopa_bcpnn_d2']['reverse'] = -1. # -1.
        self.params['params_dopa_bcpnn_d2']['gain'] = self.params['gain_d2']
        #self.params['params_dopa_bcpnn_d2']['tau_p'] = 100.
        
        
        #self.params['params_dopa_bcpnn_actions_rp'] = dc(self.params['params_dopa_bcpnn'])
        
        # ========================
        # BCPNN parameters RP / REW 
        # ========================
        #Connections States Actions
        self.params['synapse_RP'] = 'bcpnn_dopa_synapse_RP'
        self.params['params_dopa_bcpnn_RP'] = dc(self.params['params_dopa_bcpnn'])
        self.params['params_dopa_bcpnn_RP']['dopamine_modulated']= True
        self.params['params_dopa_bcpnn_RP']['p_i']= .01
        self.params['params_dopa_bcpnn_RP']['p_j']= .01
        self.params['params_dopa_bcpnn_RP']['p_ij']=  .0001
        self.params['params_dopa_bcpnn_RP']['k_pow']= 2.
        self.params['params_dopa_bcpnn_RP']['tau_i']= 5.
        self.params['params_dopa_bcpnn_RP']['tau_j']= 5.
        self.params['params_dopa_bcpnn_RP']['tau_e']= 50.
        self.params['params_dopa_bcpnn_RP']['tau_p']= 10000.
        self.params['params_dopa_bcpnn_RP']['tau_n']= 100.
        self.params['params_dopa_bcpnn_RP']['fmax']= self.params['rp_fmax']
        self.params['params_dopa_bcpnn_RP']['gain_dopa']= 1.
        self.params['params_dopa_bcpnn_RP']['gain']= self.params['gain_rp']

        if not self.params['params_dopa_bcpnn_RP']['dopamine_modulated']:
            self.params['params_dopa_bcpnn_RP'] = {'p_i': self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'], 
                'gain': - self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']} 


        # ==========================
        # BRAINSTEM
        # ==========================
        self.params['Kb'] = 1.
        self.params['gainb_neuron'] = 1.
        self.params['gainb'] = 3.
        self.params['tau_ib'] = 5.
        self.params['tau_jb'] = 6.
        self.params['tau_eb'] = 50.
        self.params['tau_pb'] = 3000.


        self.params['model_brainstem_neuron']= 'iaf_cond_alpha_bias'
        self.params['num_brainstem_neurons']= 10
        self.params['param_brainstem_neuron']= {'kappa': self.params['Kb'] ,'fmax':self.params['fmax'], 
                    'tau_j': self.params['tau_jb'],'tau_e': self.params['tau_eb'],'tau_p':self.params['tau_pb'], 
                    'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gainb_neuron']}

        self.params['synapse_states_brainstem'] = 'bcpnn_synapse'
        self.params['weight_actions_brainstem'] = 10.
        self.params['delay_actions_brainstem'] = 1.
        self.params['params_synapse_states_brainstem'] = {'p_i':self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'], 
                'gain': self.params['gainb'], 'K': self.params['Kb'], 'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':5.0,
                'tau_i': self.params['tau_ib'],'tau_j': self.params['tau_jb'],'tau_e': self.params['tau_eb'],'tau_p': self.params['tau_pb']} 


        # =========================
        # NOISE 
        # =========================
        self.params['noise_weight_d1_exc']= 1.5 
        self.params['noise_weight_d1_inh']= 1.5 
        self.params['noise_weight_d2_exc']= 1.5
        self.params['noise_weight_d2_inh']= 1.5
        self.params['noise_weight_actions_exc']= 1.5
        self.params['noise_weight_actions_inh']= 1.5
        self.params['noise_weight_str_exc']= 1.5
        self.params['noise_weight_rp_exc']= 1.5
        self.params['noise_weight_rp_inh']= 1.5
        self.params['noise_delay_d1_exc']= 1. 
        self.params['noise_delay_d1_inh']= 1. 
        self.params['noise_delay_d2_exc']= 1.
        self.params['noise_delay_d2_inh']= 1.
        self.params['noise_delay_actions_exc']= 1.
        self.params['noise_delay_actions_inh']= 1.
        self.params['noise_delay_rp_exc']= 1.
        self.params['noise_delay_rp_inh']= 1.
        self.params['noise_delay_str_exc']= 1.
        self.params['noise_rate_d1_exc']=5100.  #3000.  #5500. 
        self.params['noise_rate_d1_inh']=1000.
        self.params['noise_rate_d2_exc']=5100.  #3000. #5500.
        self.params['noise_rate_d2_inh']=1000.
        self.params['noise_rate_actions_exc']=4000.  #3600.
        self.params['noise_rate_actions_inh']=1200.  #1000.
        self.params['noise_rate_str_exc']=1200. #1000. 
        self.params['noise_rate_rp_exc']=2000.  
        self.params['noise_rate_rp_inh']=1000.  



        # =========================
        # RECORDING PARAMETERS
        # =========================

        self.params['recorded'] = 1
        self.params['prob_volt'] = .1

    def set_recorders(self):

	 pass

        

    def set_filenames(self, folder_name=None):
        """
        This function is called if no params_fn is passed 
        """

        self.set_folder_names()
        
        self.params['states_spikes_fn'] = 'states_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_spikes_fn'] = 'd1_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_volt_fn'] = 'd1_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_spikes_fn'] = 'd2_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_volt_fn'] = 'd2_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['actions_spikes_fn'] = 'actions_spikes_'
        self.params['actions_volt_fn'] = 'actions_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['efference_spikes_fn'] = 'efference_spikes_'
        self.params['rew_spikes_fn'] = 'rew_spikes_'
        self.params['rew_volt_fn'] = 'rew_volt_'
        self.params['rp_spikes_fn'] = 'rp_spikes_'
        self.params['rp_volt_fn'] = 'rp_volt_'
        
        self.params['test_rp_spikes_fn'] = 'test_rp_spikes_'
        self.params['brainstem_spikes_fn'] = 'brainstem_spikes_'
        
        self.params['states_spikes_fn_merged'] = 'states_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_spikes_fn_merged'] = 'd1_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_volt_fn_merged'] = 'd1_merged_volt.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_volt_fn_merged'] = 'd2_merged_volt.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_spikes_fn_merged'] = 'd2_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['actions_spikes_fn_merged'] = 'actions_merged_spikes.dat'
        self.params['actions_volt_fn_merged'] = 'actions_merged_volt.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['efference_spikes_fn_merged'] = 'efference_merged_spikes.dat'
        self.params['rew_spikes_fn_merged'] = 'rew_merged_spikes.dat'
        self.params['rew_volt_fn_merged'] = 'rew_merged_volt.dat'
        self.params['rp_spikes_fn_merged'] = 'rp_merged_spikes.dat'
        self.params['rp_volt_fn_merged'] = 'rp_merged_volt.dat'

        self.params['brainstem_spikes_fn_merged'] = 'brainstem_merged_spikes.dat'
        # input spike files
#        self.params['input_st_fn_mpn'] = self.params['input_folder_mpn'] + 'input_spikes_'
#        self.params['input_rate_fn_mpn'] = self.params['input_folder_mpn'] + 'input_rate_'
#        self.params['input_nspikes_fn_mpn'] = self.params['input_folder_mpn'] + 'input_nspikes_'
        # tuning properties
#        self.params['tuning_prop_exc_fn'] = self.params['parameters_folder'] + 'tuning_prop_exc.txt'
#        self.params['tuning_prop_inh_fn'] = self.params['parameters_folder'] + 'tuning_prop_inh.txt'
#        self.params['gids_to_record_fn_mp'] = self.params['parameters_folder'] + 'gids_to_record_mpn.txt'
        # storage for actions (BG), network states (MPN) and motion parameters (on Retina)
        self.params['actions_taken_fn'] = self.params['data_folder'] + 'actions_taken.txt'
        self.params['states_fn'] = self.params['data_folder'] + 'states.txt'
        self.params['rewards_fn'] = self.params['data_folder'] + 'rewards.txt'
#        self.params['motion_params_fn'] = self.params['data_folder'] + 'motion_params.txt'

        # connection filenames
        self.params['d1_conn_fn_base'] = self.params['connections_folder'] + 'd1_connections'
        self.params['d2_conn_fn_base'] = self.params['connections_folder'] + 'd2_connections'

        self.params['d1_weights_fn'] = self.params['connections_folder'] + 'd1_merged_connections.txt'
        self.params['d2_weights_fn'] = self.params['connections_folder'] + 'd2_merged_connections.txt'

        self.params['rewards_multi_fn'] = self.params['multi_folder'] + 'rewards'
        self.params['weights_d1_multi_fn'] = self.params['multi_folder'] + 'weights_d1'
        self.params['weights_d2_multi_fn'] = self.params['multi_folder'] + 'weights_d2'
        self.params['weights_rp_multi_fn'] = self.params['multi_folder'] + 'weights_rp'

    def set_folder_names(self):
    #    super(global_parameters, self).set_default_foldernames(folder_name)
    #    folder_name = 'Results_GoodTracking_titeration%d/' % self.params['t_iteration']
        folder_name = 'Test/'

#        if self.params['supervised_on'] == True:
#            folder_name += '_WithSupervisor/'
#        else:
#            folder_name += '_NoSupervisor/'
        assert(folder_name[-1] == '/'), 'ERROR: folder_name must end with a / '

        self.set_folder_name(folder_name)

        self.params['parameters_folder'] = "%sParameters/" % self.params['folder_name']
        self.params['multi_folder'] = "%sMulti/" % self.params['folder_name']
        self.params['figures_folder'] = "%sFigures/" % self.params['folder_name']
        self.params['connections_folder'] = "%sConnections/" % self.params['folder_name']
        self.params['tmp_folder'] = "%stmp/" % self.params['folder_name']
        self.params['data_folder'] = '%sData/' % (self.params['folder_name']) # for storage of analysis results etc
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['parameters_folder'], \
                            self.params['figures_folder'], \
                            self.params['tmp_folder'], \
                            self.params['connections_folder'], \
                            self.params['multi_folder'], \
                            self.params['data_folder']]

        if self.params['rank_multi']== 0:
            self.params['params_fn_json'] = '%ssimulation_parameters.json' % (self.params['parameters_folder'])


#            self.params['input_folder_mpn'] = '%sInputSpikes_MPN/' % (self.params['folder_name'])
            self.params['spiketimes_folder'] = '%sSpikes/' % self.params['folder_name']
            self.params['folder_names'].append(self.params['spiketimes_folder'])
#            self.params['folder_names'].append(self.params['input_folder_mp'])
            self.create_folders()
        else:
            self.params['params_fn_json'] = '%(folder)s%(rank)s_simulation_parameters.json' % {'folder':self.params['parameters_folder'], 'rank':str(self.params['rank_multi'])}


#            self.params['input_folder_mpn'] = '%sInputSpikes_MPN/' % (self.params['folder_name'])
            self.params['spiketimes_folder'] = '%(folder)sSpikes_%(rank)s/' % {'folder':self.params['folder_name'], 'rank':str(self.params['rank_multi'])}
            self.params['folder_names'].append(self.params['spiketimes_folder'])
#            self.params['folder_names'].append(self.params['input_folder_mp'])
            self.create_folders()

