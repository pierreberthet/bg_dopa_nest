import os
import numpy as np
import json
import ParameterContainer
from copy import deepcopy as dc
import pprint



##
# By convention, the parameters are named as, when applicable: "parameter"_"source"_"target"   or along this line.







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
        self.params['t_selection'] = 500.  # [ms]        
        self.params['t_efference'] = 250.  # [ms]
        self.params['t_reward']    = 250.  # [ms] 
        self.params['t_rest']      = 500.  # [ms] 
        self.params['t_delay']     = 0.    # [ms] 
        self.params['delay']       = False    # if true, stops the state activity earlier. 
        self.params['t_init']      = 2500. # [ms] 
        # time % (modulo) resolution has to be 0  --> for Figures processing
        self.params['resolution'] = 250.   # [ms]

        self.params['block_len'] = 40       # number of trials in a block
        self.params['n_blocks'] = 15        # number of blocks
        self.params['dt'] = .1              # [ms] /default .1

        self.params['record_spikes'] = True
        self.params['record_voltages'] = False 
        self.params['light_record'] = True
        self.params['softmax'] = True       #False = WTA

        self.params['threshold']= 0.05

        # ##############################
        # BASAL GANGLIA PARAMETERS
        # ##############################

    def set_bg_params(self):
        """
        Parameters for Basal Ganglia        
        """
        self.params['n_states']  = 3  # number of states
        self.params['n_actions'] = 3  # number of actions

        self.params['random_divconnect_poisson'] = 1.           # set to 1. if num poisson == 1, otherwise lower value? (was 0.75)
        self.params['random_connect_voltmeter'] = 0.01          # proportion of cells recorded by population specific voltmeters (too high causes OOM crashes) 
        
        self.params['epsilon'] = 0.0001 
        self.params['tau_i'] = 5.
        self.params['tau_j'] = 6.
        self.params['tau_e'] = 40.
        self.params['tau_p'] = 500. #1000.
        self.params['gain'] = 9. 
        self.params['gain_d1'] = 5.         # gain for the weights in D1
        self.params['gain_d2'] = 5.         # gain for the weights in D2
        self.params['gain_rp'] = -2.        # gain for the weights in RP
        self.params['gain_dopa'] = 25.      # gain for the plasticity amplitude
        self.params['gain_neuron'] = 20.    # gain for the neuron model has different impact (amplifies current injected) than gain in synapse model (amplifies weights)
        self.params['K'] = 0.
        self.params['fmax'] = 35.       # Cortico-matrisomal synapses, should be set to the max firing rate of the pre and post synaptic populations.
        self.params['rp_fmax'] = 15.    # Striosomo-dopaminergic synapses, should be set to the max firing rate of the pre and post synaptic populations.
        self.params['rpe'] = 0.01       # Initial value, not used 
        self.params['positive_prior'] = .01 # Initial value, not used

        ########### NEURON PARAMETERS
        self.params['Vth'] = -45.       # Spike threshold [mV] 
        self.params['Vm'] = -85.        # Membrane potential [mV]
        self.params['Cm'] = 150.        # Capacity of the membrane [pF]
        self.params['Vreset'] = -80.    # Reset potential of the membrane after spike [mV]
        self.params['gL'] = 16.6667     # Leak conductance [nS]

        self.params['temperature'] = 25.# Temperature of the softmax function

        self.params['Cm_std'] = 5.      # SD of capacity of the membrane
        self.params['Vth_std'] = 1.     # SD of the threshold value 
        self.params['Vm_std'] = 1.      # SD of the membrane potential
        self.params['Vreset_std'] = 1.  # SD of the reset value
        
        ########### SIMULATION PARAMETERS
        self.params['trigger1']= False  # Trigger of some event in the main.py simulation loop, e.g. PD neurodegenerescence
        self.params['trigger2']= False  # Trigger of some event in the main.py simulation loop, e.g. PD neurodegenerescence
        self.params['block_trigger1'] = 8 # Block in which the event will occur, if self.params['trigger1']
        self.params['block_trigger2'] = 6 # Block in which the event will occur, if self.params['trigger2']
        self.params['value_trigg_dopa_death'] = 33. # Percentage of dopamine neurons silenced by the disease

        self.params['binsize_histo_raster'] = 50.   # Spike count in GPi, used for the selection


        # ========================
        # Striatum MSN 
        # ========================

        self.params['model_state_neuron'] = 'iaf_cond_alpha'  # Neuron model

        self.params['model_exc_neuron'] = 'iaf_cond_alpha_bias' # Neuron model
        self.params['model_inh_neuron'] = 'iaf_cond_alpha_bias' # Neuron model
        self.params['num_msn_d1'] = 30      # Number of MSN D1 / action
        self.params['num_msn_d2'] = 30      # Number of MSN D2 / action
        # Parameters for the neuron model
        self.params['param_msn_d1'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm'], 'kappa': self.params['K'] ,'fmax':self.params['fmax'],'V_reset': self.params['Vreset'], 
                'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p':self.params['tau_p'], 'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gain_neuron']}
        self.params['param_msn_d2'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm'], 'kappa': self.params['K'] ,'fmax':self.params['fmax'],'V_reset': self.params['Vreset'], 
                'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p':self.params['tau_p'], 'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gain_neuron']}

        
        # ========================
        # GPi / SNr 
        # ========================
        self.params['model_gpi_neuron'] = 'iaf_cond_alpha' # Neuron model
        self.params['num_gpi'] = 10      # Number of neurons per actions in GPi/SNr
        self.params['param_gpi'] = {'V_reset': self.params['Vreset']} # to adapt parms to aif_cond_alpha neuron model
        
        self.params['str_gpi_exc_w'] = 15.  # Synaptic weight from striatal MSNs D2 to GPi/SNr
        self.params['str_gpi_inh_w'] = -3.  # Synaptic weight from striatal MSNs D1 to GPi/SNr
        self.params['str_gpi_exc_delay'] = 2. # Synaptic delay from striatal MSNs D2 to GPi/SNr
        self.params['str_gpi_inh_delay'] = 2. # Synaptic delay from striatal MSNs D1 to GPi/SNr
        self.params['std_str_gpi_exc_w'] = .2 # Standard deviation for the distribution of the MSNs D2 to GPi/SNr weights
        self.params['std_str_gpi_inh_w'] = .1 # Standard deviation for the distribution of the MSNs D1 to GPi/SNr weights
        self.params['std_str_gpi_exc_delay'] = .2 # Standard deviation for the distribution of the MSNs D2 to GPi/SNr delays
        self.params['std_str_gpi_inh_delay'] = .2 # Standard deviation for the distribution of the MSNs D1 to GPi/SNr delays

        # ========================
        # RP and REWARD 
        # ========================
        self.params['model_rp_neuron'] = 'iaf_cond_alpha_bias' # Neuron model
        self.params['num_rp_neurons'] = 15      # Number of striosomal neurons / (state-action pair)
        # Parameters of the neuron model
        self.params['param_rp_neuron'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm'], 'kappa': self.params['K'] ,'fmax':self.params['rp_fmax'], 'V_reset': self.params['Vreset'],
                'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p':self.params['tau_p'], 'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gain_neuron']}

        
        self.params['num_rew_neurons'] = 200    # Number of dopaminergic neurons
        self.params['model_rew_neuron'] = 'iaf_cond_alpha_bias' # Neuron model
        # Parameters of the neuron model
        self.params['param_rew_neuron'] = {'V_th': self.params['Vth'], 'C_m': self.params['Cm'], 'kappa': self.params['K'] ,
                                            'fmax':self.params['rp_fmax'], 'V_reset': self.params['Vreset'],
                                            'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p':self.params['tau_p'],
                                            'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gain_neuron']}

        self.params['weight_rp_rew'] = -0. # Inhibition of the dopaminergic neurons in rew by the 
                                           # current reward prediction from rp[current state, selected action]
        self.params['delay_rp_rew'] = 1.
        self.params['vt_params'] = {}


        # ========================
        # CONNECTIONS 
        # ========================
        #Connections Actions and States to RP
        # parameters for the standard bcpnn connections
        self.params['p_i'] =   .1   # np.random.random()/33.
        self.params['p_j'] =   .1   # np.random.random()/33.
        self.params['p_ij']=   .01   # self.params['p_i'] * self.params['p_j'] + np.random.random()/30.
       # self.params['p_ij']= (np.random.random()/20.)*(np.random.random()/20.)
       # print 'InitialP Pi', self.params['p_i'], 'pj',self.params['p_j'] , 'Pij', self.params['p_ij'] 
       # self.params['p_i'] = self.params['t_selection'] / ( self.params['n_states']*self.params['t_iteration']  ) 
       # self.params['p_j'] = ( self.params['t_reward'] + self.params['t_efference'] ) / ( self.params['n_actions']*self.params['t_iteration']  )
       # self.params['p_ij']= self.params['p_i'] * self.params['p_j']

        # Initial values of the P-traces and parameters for the D1 and D2 pathways
        self.params['p_i_d1']  =   self.params['t_selection'] / ( self.params['n_states']*self.params['t_iteration'] ) 
        self.params['p_j_d1']  = ( self.params['t_reward'] + self.params['t_efference'] ) / ( self.params['n_actions']*self.params['t_iteration'] )
        self.params['p_ij_d1'] =   self.params['p_i_d1'] * self.params['p_j_d1'] + self.params['positive_prior'] 
        self.params['std_p_i_d1']  = self.params['p_i_d1'] * 10./100 
        self.params['std_p_j_d1']  = self.params['p_j_d1'] * 10./100  
        self.params['std_p_ij_d1'] = self.params['p_ij_d1'] * 10./100 


        self.params['p_i_d2']  =   self.params['t_selection'] / ( self.params['n_states']*self.params['t_iteration'] ) 
        self.params['p_j_d2']  = ( self.params['t_reward'] + self.params['t_efference'] ) / ( self.params['n_actions']*self.params['t_iteration'] )
        self.params['p_ij_d2'] =   self.params['p_i_d2'] * self.params['p_j_d2'] + self.params['positive_prior'] 
        self.params['std_p_i_d2']  = self.params['p_i_d2'] * 10./100 
        self.params['std_p_j_d2']  = self.params['p_j_d2'] * 10./100  
        self.params['std_p_ij_d2'] = self.params['p_ij_d2'] * 10./100 

        
        self.params['delay_d1'] = 2.
        self.params['delay_d2'] = 2.
        self.params['std_delay_d1'] = .2
        self.params['std_delay_d2'] = .2
        




        # ========================
        #  
        # ========================
        # Initial values of the P-traces and parameters of the RP pathway
        self.params['p_ir'] = 0.01
        self.params['p_jr'] = 0.01
        self.params['p_ijr']= 0.0001

        #self.params['p_i_rp']  =   self.params['t_selection'] / ( self.params['n_states']*self.params['t_iteration'] ) 
        #self.params['p_j_rp']  =   self.params['t_efference'] / ( self.params['n_actions']*self.params['n_states']*self.params['t_iteration'] )
        self.params['p_i_rp']  =  self.params['p_ir'] 
        self.params['p_j_rp']  =  self.params['p_jr'] 
        self.params['p_ij_rp'] =   self.params['p_i_rp'] * self.params['p_j_rp'] #+ self.params['positive_prior']
        self.params['std_p_i_rp']  = self.params['p_i_rp'] * 10./100 
        self.params['std_p_j_rp']  = self.params['p_j_rp'] * 10./100  
        self.params['std_p_ij_rp'] = self.params['p_ij_rp'] * 10./100 

        self.params['delay_rp'] = 2.
        self.params['std_delay_rp'] = .1

        self.params['gpi_rp'] = 'bcpnn_dopamine_synapse' # Synapse model
        # Synapse parameters
        self.params['param_gpi_rp'] = {'p_i': self.params['p_ir'], 'p_j': self.params['p_jr'], 'p_ij': self.params['p_ijr'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']}
        self.params['states_rp'] = 'bcpnn_dopamine_synapse' # Synapse model
        # Synapse parameters
        self.params['param_states_rp'] = {'p_i': self.params['p_ir'], 'p_j': self.params['p_jr'], 'p_ij': self.params['p_ijr'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':self.params['t_selection']+self.params['t_efference'],
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']}

        self.params['bcpnn'] = 'bcpnn_synapse' # Synapse model
        # Synapse parameters
        self.params['param_bcpnn'] = {'p_i': self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'],
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']} 
        self.params['lateral_synapse_d1'] = 'bcpnn_inhib_d1'
        self.params['lateral_synapse_d2'] = 'bcpnn_inhib_d2'
        self.params['params_lateral_synapse_d1'] = {}
        self.params['params_lateral_synapse_d2'] = {}

        self.params['inhib_lateral_weights_d1'] = -4. #-2. -4.  # Synaptic weights of the lateral connections within striatum MSN D1 -> MSN D1
        self.params['inhib_lateral_weights_d2'] = -4. #-2. -4. # Synaptic weights of the lateral connections within striatum MSN D2 -> MSN D2
        self.params['inhib_lateral_weights_d2_d1'] = -1 # Synaptic weights of the lateral connections within striatum MSN D2 -> MSN D1
        self.params['inhib_lateral_delay_d1'] = 1. # Synaptic delays of the lateral connections within striatum MSN D1 -> MSN D1
        self.params['inhib_lateral_delay_d2'] = 1. # Synaptic delays of the lateral connections within striatum MSN D2 -> MSN D2
        self.params['inhib_lateral_delay_d2_d1'] = 1. # Synaptic delays of the lateral connections within striatum MSN D2 -> MSN D1
        self.params['std_inhib_lateral_weights_d1'] = .01 #-4. # Standard deviation of the distribution of the synaptic weights MSN D1 -> MSN D1
        self.params['std_inhib_lateral_weights_d2'] = .01 #-4. # Standard deviation of the distribution of the synaptic weights MSN D2 -> MSN D2
        self.params['std_inhib_lateral_weights_d2_d1']= .01 # Standard deviation of the distribution of the synaptic weights MSN D2 -> MSN D1
        self.params['std_inhib_lateral_delay_d1'] = .1 # Standard deviation of the distribution of the delays MSN D1 -> MSN D1
        self.params['std_inhib_lateral_delay_d2'] = .1 # Standard deviation of the distribution of the delays MSN D2 -> MSN D2
        self.params['std_inhib_lateral_delay_d2_d1']= .1 #Standard deviation of the distribution of the delays MSN D2 -> MSN D1
        self.params['ratio_lat_inh_d1_d1'] = 2.   # ratio of D1 MSNs belonging to the other actions inhibited by one D1 MSN from a specific action  
        self.params['ratio_lat_inh_d2_d2'] = 2.   # ratio of D2 MSNs belonging to the other actions inhibited by one D2 MSN from a specific action
        self.params['ratio_lat_inh_d2_d1'] = 3.   # ratio of D1 MSNs inhibited by the D2 MSN from the same action

        # ========================
        # Dopa BCPNN parameters 
        # ========================
        #Connections States Actions
        
        self.params['synapse_d1'] = 'bcpnn_dopamine_synapse_d1' # Synapse model
        # Synapse parameters
        self.params['params_synapse_d1'] = {'p_i': self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']} 
        self.params['synapse_d2'] = 'bcpnn_dopamine_synapse_d2' # Synapse model
        # Synapse parameters
        self.params['params_synapse_d2'] = {'p_i': self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'], 
                'gain': self.params['gain'], 'K': self.params['K'],'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':1.0,
                'tau_i': self.params['tau_i'],'tau_j': self.params['tau_j'],'tau_e': self.params['tau_e'],'tau_p': self.params['tau_p']} 

        # ========================
        # Volume Transmitter parameters 
        # ========================

        self.params['w_rew_vtdopa'] = 1. # Synaptic weights of the connections from the dopaminergic neurons to the volume transmitter
        self.params['delay_rew_vtdopa'] = 2. # Synaptic delay of the connections from the dopaminergic neurons to the volume transmitter
        self.params['std_w_rew_vtdopa'] = .01 # Standard deviation of the distribution of the weights form dopa neurons to the volume transmitter
        self.params['std_delay_rew_vtdopa'] = .1 # Standard deviation of the distribution of the delays form dopa neurons to the volume transmitter


        # ========================
        # Spike DETECTORS parameters
        # ========================

        self.params['spike_detector_gpi'] = {"withgid":True, "withtime":True}
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


        self.params['num_neuron_poisson_efference'] = 1     # Number or Poisson generators for the efference copy
        self.params['num_neuron_poisson_input_BG'] =1       # Number of Poisson generators for the states
        self.params['active_full_efference_rate'] = 800.    # Rate of the Poisson generator for the active action
        self.params['inactive_efference_rate'] = 1.         # Rate of the Poisson generator for the inactive actions
        self.params['active_poisson_input_rate'] = 1950.    # Rate of the Poisson generator for the active state 
        self.params['inactive_poisson_input_rate'] = 1500.  # Rate of the Poisson generator for the inactive states

        self.params['active_poisson_rew_rate'] = 3150.
        self.params['baseline_poisson_rew_rate'] = 2950.
        self.params['inactive_poisson_rew_rate'] = 2750.  #2300.
        
        #Initialisation####################
        self.params['initial_poisson_input_rate'] = 1700.  #2500.
        self.params['initial_noise_striatum_rate'] = 2500.  #2500.

        self.params['param_poisson_pop_input_BG'] = {}
        self.params['param_poisson_efference'] = {}
        
        self.params['model_poisson_rew'] = 'poisson_generator'
        self.params['num_poisson_rew'] = 1
        self.params['weight_poisson_rew'] = 5.
        self.params['delay_poisson_rew'] = 1.
        self.params['std_weight_poisson_rew'] = .01
        self.params['std_delay_poisson_rew'] = .01
        self.params['param_poisson_rew'] = {}# to adapt parms to aif_cond_alpha neuron model

        self.params['weight_efference_strd1_exc'] = 4.      ## 4.
        self.params['weight_efference_strd1_inh'] = -3.     ## -2.
        self.params['weight_efference_strd2_exc'] = 4.
        self.params['weight_efference_strd2_inh'] = -3.
        self.params['std_weight_efference_strd1_exc'] = .1     ## 4.
        self.params['std_weight_efference_strd1_inh'] = .1    ## -2.
        self.params['std_weight_efference_strd2_exc'] = .1
        self.params['std_weight_efference_strd2_inh'] = .1
        self.params['delay_efference_strd1_exc'] = 2.
        self.params['delay_efference_strd1_inh'] = 2.
        self.params['delay_efference_strd2_exc'] = 2.
        self.params['delay_efference_strd2_inh'] = 2.
        self.params['std_delay_efference_strd1_exc'] = .2
        self.params['std_delay_efference_strd1_inh'] = .2
        self.params['std_delay_efference_strd2_exc'] = .2
        self.params['std_delay_efference_strd2_inh'] = .2

        self.params['weight_poisson_input'] = 4.
        self.params['delay_poisson_input'] = 1.
        
        self.params['num_neuron_states'] = 30
        self.params['param_states_pop'] = {} 

        self.params['weight_states_rp']    = 6.  #3.
        self.params['weight_efference_rp'] = 5.5
        #self.params['weight_states_rp']    = 0.  #3.
        #self.params['weight_efference_rp'] = 0.
        if self.params['delay']:
            self.params['delay_states_rp']    =  self.params['t_efference']
            self.params['delay_efference_rp'] =  self.params['t_efference']
        else:
            self.params['delay_states_rp']    =  2.
            self.params['delay_efference_rp'] =  self.params['t_efference'] #2.

        self.params['std_weight_states_rp']    = .01  
        self.params['std_weight_efference_rp'] = .01
        #self.params['std_weight_states_rp']    = .00001  
        #self.params['std_weight_efference_rp'] = .00001
        self.params['std_delay_states_rp']    = .1
        self.params['std_delay_efference_rp'] = .1
       # self.params['weight_gpi_rp'] = 5.
       # self.params['delay_gpi_rp'] =  self.params['t_efference']

        # ========================
        # Dopa BCPNN parameters 
        # ========================
        

        self.params['dopa_b'] = -.054 #-.052 #-.077  #- .0697   ###.069 #-.085  #-1.4  #-0.13   # - (baseline rate dopa (= pop size * rate) )/ 1000
        self.params['weight'] = 5.

        self.params['dopa_bcpnn'] = 'bcpnn_dopamine_synapse'
        self.params['params_dopa_bcpnn'] ={
            'bias':0.0,   #ANN interpretation. Only calculated here to demonstrate match to rule. 
                          # Will be eliminated in future versions, where bias will be calculated postsynaptically       
            'b':self.params['dopa_b'],
            'delay':1.,
            'dopamine_modulated':True,
            'complementary':False, #set to use the complementary traces or not
            'positive_only':True,
            'e_i':0.01,
            'e_j':0.01,
            'e_j_c':.99,
            'e_ij':0.001,
            'e_ij_c':0.3,
            'epsilon':0.001, #lowest possible probability of spiking, e.g. lowest assumed firing rate
            'fmax':self.params['fmax'],    #Frequency assumed as maximum firing, for match with abstract rule
            'gain':self.params['gain'],    #Coefficient to scale weight as conductance, can be zero-ed out
            'gain_dopa':self.params['gain_dopa'], 
            'k_pow': 5.,
            'K':0.,         #Print-now signal // Neuromodulation. Turn off learning, K = 0
            'sigmoid':0.,
            'sigmoid_mean':0.,
            'sigmoid_slope':1.,
            'n': .07, #17,
            'p_i':   self.params['p_i'],    #.01,        #0.01,
            'p_j':   self.params['p_j'],    #.01,       #0.01,
            'p_ij':  self.params['p_ij'],   #.0001,     #0.0001,
            'reverse':1., #1. 
            'tau_i':self.params['tau_i'],     #Primary trace presynaptic time constant
            'tau_j':self.params['tau_j'],      #Primary trace postsynaptic time constant
            'tau_e':self.params['tau_e'],      #Secondary trace time constant
            'tau_p':self.params['tau_p'],     #Tertiarty trace time constant
            'tau_n':200.,    #default 100
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
        self.params['params_dopa_bcpnn_RP']['positive_only'] = False 
        self.params['params_dopa_bcpnn_RP']['dopamine_modulated']= True
        self.params['params_dopa_bcpnn_RP']['p_i']= .01
        self.params['params_dopa_bcpnn_RP']['p_j']= .01
        self.params['params_dopa_bcpnn_RP']['p_ij']=  .0001 #.0001051271096376 #.0001  Value to get an initial weight of 0.05
        self.params['params_dopa_bcpnn_RP']['k_pow']= 2.  #2.
        self.params['params_dopa_bcpnn_RP']['tau_i']= 12.
        self.params['params_dopa_bcpnn_RP']['tau_j']= 15.
        self.params['params_dopa_bcpnn_RP']['tau_e']= 50.
        self.params['params_dopa_bcpnn_RP']['tau_p']= 2000.
        self.params['params_dopa_bcpnn_RP']['tau_n']= 100.
        self.params['params_dopa_bcpnn_RP']['fmax']= self.params['rp_fmax']
        self.params['params_dopa_bcpnn_RP']['gain_dopa']= 25. #8. #14.
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
        self.params['gainb_neuron'] = 20.
        self.params['gainb'] = 4.
        self.params['tau_ib'] = 10.
        self.params['tau_jb'] = 20.
        self.params['tau_eb'] = 100.
        self.params['tau_pb'] = 2000.


        self.params['model_brainstem_neuron']= 'iaf_cond_alpha_bias'
        self.params['num_brainstem_neurons']= 20
        self.params['param_brainstem_neuron']= {'kappa': self.params['Kb'] ,'fmax':self.params['fmax'], 
                    'tau_j': self.params['tau_jb'],'tau_e': self.params['tau_eb'],'tau_p':self.params['tau_pb'], 
                    'epsilon': self.params['epsilon'], 't_ref': 2.0, 'gain': self.params['gainb_neuron']}

        self.params['synapse_states_brainstem'] = 'bcpnn_synapse'
        self.params['weight_gpi_brainstem'] = -4. #-7.
        self.params['delay_gpi_brainstem'] = 1.
        self.params['self_exc_bs'] = 4. 
        self.params['delay_self_exc_bs'] = 1.
        self.params['lat_inh_bs'] = -10.
        self.params['delay_lat_inh_bs'] = 1.


        self.params['params_synapse_states_brainstem'] = {'p_i':self.params['p_i'], 'p_j': self.params['p_j'], 'p_ij': self.params['p_ij'], 
                'gain': self.params['gainb'], 'K': self.params['Kb'], 'fmax': self.params['fmax'],
                'epsilon': self.params['epsilon'],'delay':5.0,
                'tau_i': self.params['tau_ib'],'tau_j': self.params['tau_jb'],'tau_e': self.params['tau_eb'],'tau_p': self.params['tau_pb']} 


        # =========================
        # NOISE 
        # =========================
        self.params['noise_weight_d1_exc']= 5.5# 1. 
        self.params['noise_weight_d1_inh']= 2.# 1. 
        self.params['noise_weight_d2_exc']= 5.5# 1.
        self.params['noise_weight_d2_inh']= 2.# 1.
        self.params['noise_weight_gpi_exc']= 2.5 
        self.params['noise_weight_gpi_inh']= 1.5
        self.params['noise_weight_str_exc']= 1.5
        self.params['noise_weight_rp_exc']= 2.
        self.params['noise_weight_rp_inh']= 1.5
        self.params['noise_weight_bs_exc']= 3.
        self.params['noise_weight_bs_inh']= 2.
        self.params['noise_delay_d1_exc']= 1. 
        self.params['noise_delay_d1_inh']= 1. 
        self.params['noise_delay_d2_exc']= 1.
        self.params['noise_delay_d2_inh']= 1.
        self.params['noise_delay_gpi_exc']= 1.
        self.params['noise_delay_gpi_inh']= 1.
        self.params['noise_delay_rp_exc']= 1.
        self.params['noise_delay_rp_inh']= 1.
        self.params['noise_delay_bs_exc']= 1.
        self.params['noise_delay_bs_inh']= 1.
        self.params['noise_delay_str_exc']= 1.
        self.params['noise_rate_d1_exc']=2200.  #3000.  #5500. 
        self.params['noise_rate_d1_inh']=1000.
        self.params['noise_rate_d2_exc']=2200.  #3000. #5500.
        self.params['noise_rate_d2_inh']=1000.
        self.params['noise_rate_gpi_exc']=3400.  #3400
        self.params['noise_rate_gpi_inh']=1200.  #1200.
        self.params['noise_rate_str_exc']=1000. #1000. 
        self.params['noise_rate_rp_exc']=2500.  
        self.params['noise_rate_rp_inh']=1000.  
        self.params['noise_rate_bs_exc']=7500. #5500.  
        self.params['noise_rate_bs_inh']=1000.  



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

