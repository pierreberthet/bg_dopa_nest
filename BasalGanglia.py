import numpy as np
import nest
import utils
from scipy.stats import itemfreq
import pprint

pp = pprint.PrettyPrinter(indent=4)

class BasalGanglia(object):

    def __init__(self, params, comm=None):

        self.params = params
        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.comm = comm # mpi communicator needed to broadcast nspikes between processes
        if comm != None:
            assert (comm.rank == self.pc_id), 'mpi4py and NEST tell me different PIDs!'
            assert (comm.size == self.n_proc), 'mpi4py and NEST tell me different PIDs!'

        pyrngs = [np.random.RandomState(s) for s in xrange(self.comm.size)]

        self.strD1 = {}
        self.strD2 = {}
        self.actions = {}
        self.vt_dopa = nest.Create('volume_transmitter', 1, self.params['vt_params'])
        self.rp = {}
        self.efference_copy = {}

        self.who = self.params['recorded']
        self.rec_count = 0

        self.recorder_output= {}
        self.recorder_output_gidkey = {}
        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder'], 'overwrite_files': True})
        
        if self.params['record_spikes']:

            # Recording devices
            self.recorder_d1 = {}
            self.recorder_d2 = {}
            self.recorder_states = {}
            self.recorder_efference = {}
            self.recorder_rp = {}
            self.recorder_rew = nest.Create("spike_detector", params= self.params['spike_detector_rew'])
            nest.SetStatus(self.recorder_rew,[{"to_file": True, "withtime": True, 'label' : self.params['rew_spikes_fn']}])

            #self.recorder_test_rp = nest.Create("spike_detector", params= self.params['spike_detector_test_rp'])
            #nest.SetStatus(self.recorder_test_rp, [{"to_file": True, "withtime": True, 'label' : self.params['test_rp_spikes_fn']}])
        if self.params['record_voltages']:
            self.voltmeter_rp = {}
            self.voltmeter_rew = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
            nest.SetStatus(self.voltmeter_rew, [{"to_file": True, "withtime": True, 'label' : self.params['rew_volt_fn']}])
            self.voltmeter_action = {}
            self.voltmeter_d1 = {}
            self.voltmeter_d2 = {}
            


        self.t_current = self.params['t_init'] 
        


        self.create_brainstem()

        
        
        # ##########
        # REWARD 
        # ##########
        # Creates the REWARD population and its poisson input and the RP population and then connects theses different populations.
        self.rew = nest.Create( self.params['model_rew_neuron'], self.params['num_rew_neurons'], params= self.params['param_rew_neuron'] )
        self.poisson_rew = nest.Create( self.params['model_poisson_rew'], self.params['num_poisson_rew'], params=self.params['param_poisson_rew'] )
        nest.DivergentConnect(self.poisson_rew, self.rew, weight=self.params['weight_poisson_rew'], delay=self.params['delay_poisson_rew'])

#        print 'NESTCONN'
#        print 'NUM REW = ' , self.params['num_rew_neurons']
#        print nest.GetConnections(self.poisson_rew, self.rew)

        if self.params['record_spikes']:
            nest.ConvergentConnect(self.rew, self.recorder_rew)
        if self.params['record_voltages']:
            nest.ConvergentConnect(self.voltmeter_rew, np.random.choice(self.rew, int( self.params['prob_volt']*self.params['num_rew_neurons']) ))

        # Connect the dopaminergic neurons to a volume transmitter. This vt will be used as modulator in the dopa bcpnn synapses. (this dopa to vt connect has to be done before creating dopa bcpnn synapses)
        nest.ConvergentConnect(self.rew, self.vt_dopa, weight=self.params['w_rew_vtdopa'], delay =self.params['delay_rew_vtdopa'])


        # ################
        # EFFERENCE COPY
        # ################ 
        #Creates and connects the EFFERENCE COPY population.
        #This actives the D1 population coding for the selected action and the D2 populations of non-selected actions, in STR
        
        for nactions in xrange(self.params['n_actions']):
            self.efference_copy[nactions] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_efference'], params = self.params['param_poisson_efference']  )
            if self.params['record_spikes']:
                self.recorder_efference[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_efference'])
                nest.SetStatus(self.recorder_efference[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['efference_spikes_fn']+ str(nactions)}])
                nest.ConvergentConnect(self.efference_copy[nactions], self.recorder_efference[nactions])


        # #########
        # ACTIONS
        # #########

        # Creates the output ACTIONS populations, and then create the Connections with STR
        for nactions in xrange(self.params['n_actions']):
            self.actions[nactions] = nest.Create(self.params['model_bg_output_neuron'], self.params['num_actions_output'], params= self.params['param_bg_output'])
            if self.params['record_voltages']:
                self.voltmeter_action[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
                nest.SetStatus(self.voltmeter_action[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['actions_volt_fn']+ str(nactions)}])
            self.recorder_output[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_action'])
            for ind in xrange(self.params['num_actions_output']):
                self.recorder_output_gidkey[self.actions[nactions][ind]] = nactions
            nest.SetStatus(self.recorder_output[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['actions_spikes_fn']+ str(nactions)}])
            nest.ConvergentConnect(self.actions[nactions], self.recorder_output[nactions])
            if self.params['record_voltages']:
                nest.ConvergentConnect(self.voltmeter_action[nactions], np.random.choice(self.actions[nactions], int(self.params['prob_volt']*self.params['num_actions_output'])))

        print 'debug recorder gid', self.recorder_output_gidkey
        

        # ##########
        # RP / Striosomes
        # ##########
        for index_rp in xrange(self.params['n_actions'] * self.params['n_states']):
            self.rp[index_rp] = nest.Create(self.params['model_rp_neuron'], self.params['num_rp_neurons'], params= self.params['param_rp_neuron'] )
           # nest.ConvergentConnect(self.recorder_test_rp, self.rp[index_rp])
            if self.params['record_spikes']:
                self.recorder_rp[index_rp] = nest.Create("spike_detector", params= self.params['spike_detector_rp'])
                nest.SetStatus(self.recorder_rp[index_rp],[{"to_file": True, "withtime": True, 'label' : self.params['rp_spikes_fn']+ str(index_rp)}])
                nest.ConvergentConnect(self.rp[index_rp],self.recorder_rp[index_rp])
            if self.params['record_voltages']:
                self.voltmeter_rp[index_rp] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
                nest.SetStatus(self.voltmeter_rp[index_rp], [{"to_file": True, "withtime": True, 'label' : self.params['rp_volt_fn']+ str(index_rp)}])
                nest.ConvergentConnect(self.voltmeter_rp[index_rp], self.rp[index_rp])


        # #####################
        # STRIATUM MSN D1 + D2
        # ##################### 
        #Creates D1 and D2 populations in STRIATUM, connections are created later
        for nactions in range(self.params['n_actions']):
            if self.params['record_spikes']:
                self.recorder_d1[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_d1'])
                self.recorder_d2[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_d2'])
                nest.SetStatus(self.recorder_d1[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d1_spikes_fn']+ str(nactions)}])
                nest.SetStatus(self.recorder_d2[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d2_spikes_fn']+ str(nactions)}])
            # D1
            self.strD1[nactions] = nest.Create(self.params['model_exc_neuron'], 
                    self.params['num_msn_d1'], params= self.params['param_msn_d1'])
            nodes_info = nest.GetStatus(self.strD1[nactions])
            local_nodes = [(ni['global_id'], ni['vp']) for ni in nodes_info if ni['local']]
            for gid, vp in local_nodes:
                nest.SetStatus([gid], {'C_m': pyrngs[vp].normal(self.params['Cm'], self.params['Cm_std']),'V_th': pyrngs[vp].normal(self.params['Vth'], self.params['Vth_std']), 'V_reset': pyrngs[vp].normal(self.params['Vreset'], self.params['Vreset_std']) })
                #nest.SetStatus([gid], {'C_m': pyrngs[vp].normal(self.params['Cm'], self.params['Cm_std'])})
                #nest.SetStatus([gid], {'V_th': pyrngs[vp].normal(self.params['Vth'], self.params['Vth_std'])})
                #nest.SetStatus([gid], {'V_reset': pyrngs[vp].normal(self.params['Vreset'], self.params['Vreset_std'])})
            # D2
            self.strD2[nactions] = nest.Create(self.params['model_inh_neuron'], 
                    self.params['num_msn_d2'], params= self.params['param_msn_d2'])
            nodes_info = nest.GetStatus(self.strD2[nactions])
            local_nodes = [(ni['global_id'], ni['vp']) for ni in nodes_info if ni['local']]
            for gid, vp in local_nodes:
                nest.SetStatus([gid], {'C_m': pyrngs[vp].normal(self.params['Cm'], self.params['Cm_std']),'V_th': pyrngs[vp].normal(self.params['Vth'], self.params['Vth_std']), 'V_reset': pyrngs[vp].normal(self.params['Vreset'], self.params['Vreset_std']) })
                #nest.SetStatus([gid], {'V_th': pyrngs[vp].normal(self.params['Vth'], self.params['Vth_std'])})
                #nest.SetStatus([gid], {'V_reset': pyrngs[vp].normal(self.params['Vreset'], self.params['Vreset_std'])})

            if self.params['record_voltages']:
            # Recorders
                self.voltmeter_d1[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
                nest.SetStatus(self.voltmeter_d1[nactions],[{"to_file": True, 
                    "withtime": True, 'label' : self.params['d1_volt_fn']+ str(nactions)}])
                self.voltmeter_d2[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
                nest.SetStatus(self.voltmeter_d2[nactions],[{"to_file": True, 
                    "withtime": True, 'label' : self.params['d2_volt_fn']+ str(nactions)}])
                
                nest.ConvergentConnect(self.voltmeter_d1[nactions], np.random.choice(self.strD1[nactions], int(self.params['prob_volt']*self.params['num_msn_d1'])))
                nest.ConvergentConnect(self.voltmeter_d2[nactions], np.random.choice(self.strD2[nactions], int(self.params['prob_volt']*self.params['num_msn_d2'])))
            if self.params['record_spikes']:
                nest.ConvergentConnect(self.strD1[nactions], self.recorder_d1[nactions])
                nest.ConvergentConnect(self.strD2[nactions], self.recorder_d2[nactions])

        
        # #################################
        # CROSS INHIBITION STRIATUM D1 D2
        # #################################
        for nactions in range(self.params['n_actions']):
            for other in xrange(self.params['n_actions']):
                if other != nactions:
                    #nest.DivergentConnect(self.strD1[nactions], self.strD1[other], model=self.params['lateral_synapse_d1']  )
                    #nest.DivergentConnect(self.strD2[nactions], self.strD2[other], model=self.params['lateral_synapse_d2']  )
                    nest.RandomDivergentConnect(self.strD1[nactions], self.strD1[other], int(self.params['num_msn_d1']/self.params['ratio_lat_inh_d1_d1']) , weight = np.round(np.random.normal(self.params['inhib_lateral_weights_d1'], self.params['std_inhib_lateral_weights_d1'], int(self.params['num_msn_d1']/self.params['ratio_lat_inh_d1_d1'])),1).tolist(), delay= np.round(np.random.normal(self.params['inhib_lateral_delay_d1'],self.params['std_inhib_lateral_delay_d1'], int(self.params['num_msn_d1']/self.params['ratio_lat_inh_d1_d1'])),1).tolist() )
                    nest.RandomDivergentConnect(self.strD2[nactions], self.strD2[other], int(self.params['num_msn_d2']/self.params['ratio_lat_inh_d2_d2']) , weight = np.round(np.random.normal(self.params['inhib_lateral_weights_d2'], self.params['std_inhib_lateral_weights_d2'], int(self.params['num_msn_d2']/self.params['ratio_lat_inh_d2_d2'])),1).tolist(), delay= np.round(np.random.normal(self.params['inhib_lateral_delay_d2'],self.params['std_inhib_lateral_delay_d2'], int(self.params['num_msn_d2']/self.params['ratio_lat_inh_d2_d2'])),1).tolist() )
                    #nest.RandomDivergentConnect(self.strD2[nactions], self.strD2[other], int(self.params['num_msn_d2']/self.params['ratio_lat_inh_d2_d2']), weight = self.params['inhib_lateral_weights_d2'], delay= self.params['inhib_lateral_delay_d2'])
                
                #nest.RandomDivergentConnect(self.strD2[nactions], self.strD1[nactions], int(self.params['num_msn_d1']/4.), weight = self.params['inhib_lateral_weights_d2_d1'], delay= self.params['inhib_lateral_delay_d2'])
        
        # #####################
        # STATES
        # ##################### 
        #Creates input populations
        """
        Creates the inputs populations, and their respective poisson pop, and connect them to Striatum MSNs D1 D2 populations
        """
        self.states = {}
        self.input_poisson = {}
        for nstates in range(self.params['n_states']):
            self.input_poisson[nstates] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_input_BG'], params = self.params['param_poisson_pop_input_BG']  )
            self.states[nstates] = nest.Create( self.params['model_state_neuron'], self.params['num_neuron_states'], params = self.params['param_states_pop']  )
            nest.DivergentConnect(self.input_poisson[nstates], self.states[nstates], weight=self.params['weight_poisson_input'], delay=self.params['delay_poisson_input'])

            if self.params['record_spikes']:
                self.recorder_states[nstates] = nest.Create("spike_detector", params= self.params['spike_detector_states'])
                nest.SetStatus(self.recorder_states[nstates],[{"to_file": True, "withtime": True, 'label' : self.params['states_spikes_fn'] + str(nstates)}])
                nest.ConvergentConnect(self.states[nstates], self.recorder_states[nstates])
     




                                            # ############################### #
                                            #           CONNECTIONS           #
                                            # ############################### #

        self.connect_brainstem()
        self.connect_bcpnn_sensorimotor()

        # ################### #
        # STATES ACTIONS / RP #
        # ################### #
        for istate in xrange(self.params['n_states']):
            for iaction in range(self.params['n_actions']):
               # nest.SetDefaults( self.params['dopa_bcpnn'], params= self.params['params_dopa_bcpnn_actions_rp'])
                #nest.DivergentConnect(self.strD1[iaction], self.rp[iaction+istate*self.params['n_actions']], weight=self.params['weight_actions_rp'], delay=self.params['delay_actions_rp'])
                nest.DivergentConnect(self.efference_copy[iaction], self.rp[iaction+istate*self.params['n_actions']], weight=np.round(np.random.normal(self.params['weight_efference_rp'], self.params['std_weight_efference_rp'], self.params['num_rp_neurons']), 1).tolist(), delay=np.round(np.random.normal(self.params['delay_efference_rp'], self.params['std_delay_efference_rp'], self.params['num_rp_neurons']), 1).tolist())
               # print 'DATADATA',  nest.GetStatus(nest.GetConnections(self.actions[iaction], self.rp[iaction+istate*self.params['n_actions']]))        
               # nest.SetDefaults( self.params['dopa_bcpnn'], params= self.params['params_dopa_bcpnn_states_rp'])
                nest.DivergentConnect(self.states[istate], self.rp[iaction + istate*self.params['n_actions']], weight=np.round(np.random.normal(self.params['weight_states_rp'], self.params['std_weight_states_rp'],self.params['num_rp_neurons']), 1).tolist(), delay=np.round(np.random.normal(self.params['delay_states_rp'],self.params['std_delay_states_rp'],self.params['num_rp_neurons']), 1 ).tolist())
        
        # ################### #
        # STRIATUM / ACTIONS  #
        # ################### #
        for nactions in xrange(self.params['n_actions']):
            nest.ConvergentConnect(self.strD1[nactions], self.actions[nactions], weight=np.round(np.random.normal(self.params['str_to_output_inh_w'],self.params['std_str_to_output_inh_w'],self.params['num_msn_d1']),1).tolist(), delay=np.round(np.random.normal(self.params['str_to_output_inh_delay'],self.params['std_str_to_output_inh_delay'], self.params['num_msn_d1']),1).tolist()) 
            nest.ConvergentConnect(self.strD2[nactions], self.actions[nactions], weight=np.round(np.random.normal(self.params['str_to_output_exc_w'],self.params['std_str_to_output_inh_w'],self.params['num_msn_d2']),1).tolist(), delay=np.round(np.random.normal(self.params['str_to_output_exc_delay'],self.params['std_str_to_output_exc_delay'], self.params['num_msn_d2']),1).tolist())	
            # for neuron in self.actions[nactions]:
            #     nest.ConvergentConnect(self.strD1[nactions], [neuron], weight=self.params['str_to_output_inh_w'], delay=self.params['str_to_output_inh_delay']) 
            #     nest.ConvergentConnect(self.strD2[nactions], [neuron], weight=self.params['str_to_output_exc_w'], delay=self.params['str_to_output_exc_delay'])	
        # ################### #
        #   EFFERENCE / STR   #
        # ################### #
        for nactions in xrange(self.params['n_actions']):
            for i in xrange(self.params['n_actions']):
                 if i != nactions:
                     nest.DivergentConnect(self.efference_copy[nactions], self.strD1[i], weight=np.round(np.random.normal(self.params['weight_efference_strd1_inh'],self.params['std_weight_efference_strd1_inh'], self.params['num_msn_d1']),1).tolist(), delay=np.round(np.random.normal(self.params['delay_efference_strd1_inh'],self.params['std_delay_efference_strd1_inh'], self.params['num_msn_d1']),1).tolist() )
                     nest.DivergentConnect(self.efference_copy[nactions], self.strD2[i], weight=np.round(np.random.normal(self.params['weight_efference_strd2_inh'],self.params['std_weight_efference_strd2_inh'], self.params['num_msn_d2']),1).tolist(), delay=np.round(np.random.normal(self.params['delay_efference_strd2_inh'],self.params['std_delay_efference_strd2_inh'], self.params['num_msn_d2']),1).tolist() )
                     #nest.DivergentConnect(self.efference_copy[nactions], self.strD2[i], weight=self.params['weight_efference_strd2_inh'], delay=self.params['delay_efference_strd2_inh'])
            nest.DivergentConnect(self.efference_copy[nactions], self.strD1[nactions], weight=np.round(np.random.normal(self.params['weight_efference_strd1_exc'],self.params['std_weight_efference_strd1_exc'], self.params['num_msn_d1']),1).tolist(), delay=np.round(np.random.normal(self.params['delay_efference_strd1_exc'], self.params['std_delay_efference_strd1_exc'], self.params['num_msn_d1']),1).tolist())
            nest.DivergentConnect(self.efference_copy[nactions], self.strD2[nactions], weight=np.round(np.random.normal(self.params['weight_efference_strd2_exc'],self.params['std_weight_efference_strd2_exc'], self.params['num_msn_d2']),1).tolist(), delay=np.round(np.random.normal(self.params['delay_efference_strd2_exc'], self.params['std_delay_efference_strd2_exc'], self.params['num_msn_d2']),1).tolist())



                                  # ############################################ #
                                  # ############  BCPNN CONNECTIONS ############ #
                                  # ############################################ #

        #nest.SetDefaults(self.params['dopa_bcpnn'], { 'vt': self.vt_dopa[0] } ) 
        #self.create_bcpnn_sensorimotor()
        #print 'VTDOPA : ', nest.GetStatus(self.vt_dopa)
       
        nest.CopyModel('bcpnn_synapse', self.params['lateral_synapse_d1'], self.params['params_lateral_synapse_d1'])
        nest.CopyModel('bcpnn_synapse', self.params['lateral_synapse_d2'], self.params['params_lateral_synapse_d2'])
       
        # ####################################### #
        #       RP - REW // STRIOSOMES - DOPA     #
        # ####################################### #
        nest.CopyModel('bcpnn_dopamine_synapse',self.params['synapse_RP'], self.params['params_dopa_bcpnn_RP'] )
        nest.SetDefaults(self.params['synapse_RP'], { 'vt': self.vt_dopa[0] } ) 
        #nest.CopyModel('bcpnn_synapse',self.params['synapse_RP'], self.params['params_bcpnn_RP'] )
        # Creates RP populations and the connections from states and actions to the corresponding RP populations

        for index_rp in xrange(self.params['n_actions'] * self.params['n_states']):
            nest.DivergentConnect( self.rp[index_rp], self.rew, model = self.params['synapse_RP']  )
            conn = nest.GetConnections(source=self.rp[index_rp], target=self.rew, synapse_model=self.params['synapse_RP'])
            delay_params = [{'delay':np.round(np.random.normal(self.params['delay_rp'], self.params['std_delay_rp']),1)} for c in conn]
            nest.SetStatus(conn, delay_params)
            pi_params = [{'p_i':max(self.params['positive_prior'], np.round(np.random.normal(self.params['p_i_rp'], self.params['std_p_i_rp']),1))} for c in conn]
            pj_params = [{'p_j':max(self.params['positive_prior'], np.round(np.random.normal(self.params['p_j_rp'], self.params['std_p_j_rp']),1))} for c in conn]
            #pij_params = [{'p_ij':max(self.params['positive_prior']*self.params['positive_prior']+self.params['positive_prior'], np.round(np.random.normal(self.params['p_ij_rp'], self.params['std_p_ij_rp']),1))} for c in conn]
            pij_params = [{'p_ij':max(self.params['epsilon'], np.round(np.random.normal(self.params['p_ij_rp'], self.params['std_p_ij_rp']),1))} for c in conn]
            nest.SetStatus(conn, pi_params)
            nest.SetStatus(conn, pj_params)
            nest.SetStatus(conn, pij_params)
       # nodes_info = nest.GetStatus( nest.GetConnections(self.rp, self.rew))
       # local_nodes = [(ni['global_id'], ni['vp']) for ni in nodes_info if ni['local']]
       # for gid, vp in local_nodes:
       #     nest.SetStatus([gid], {'p_i': pyrngs[vp].normal(self.params['p_i'], self.params['p_i_std'])})

        # ############################################# #
        #      STATES - ACTIONS // CORTEX - STRIATUM    #
        # ############################################# #
        nest.CopyModel('bcpnn_dopamine_synapse',self.params['synapse_d1'], self.params['params_dopa_bcpnn_d1'] )
        nest.SetDefaults(self.params['synapse_d1'], { 'vt': self.vt_dopa[0] } ) 
        nest.CopyModel('bcpnn_dopamine_synapse',self.params['synapse_d2'], self.params['params_dopa_bcpnn_d2'] )
        nest.SetDefaults(self.params['synapse_d2'], { 'vt': self.vt_dopa[0] } ) 
        print 'Geronimod1 ', self.comm.rank, 'params',nest.GetDefaults(self.params['synapse_d1'])
        print 'Geronimod2 ', self.comm.rank, 'params',nest.GetDefaults(self.params['synapse_d2'])
        for nstates in range(self.params['n_states']):
            for nactions in range(self.params['n_actions']):
                # D1
                #nest.SetDefaults(self.params['dopa_bcpnn'], params=self.params['params_dopa_bcpnn_d1'])
                nest.DivergentConnect(self.states[nstates], self.strD1[nactions], model=self.params['synapse_d1'])
                conn = nest.GetConnections(source=self.states[nstates], target=self.strD1[nactions], synapse_model=self.params['synapse_d1'])
                delay_params = [{'delay':np.round(np.random.normal(self.params['delay_d1'], self.params['std_delay_d1']),1)} for c in conn]
                nest.SetStatus(conn, delay_params)
                pi_params = [{'p_i':max(self.params['epsilon'], np.round(np.random.normal(self.params['p_i_d1'], self.params['std_p_i_d1']),1))} for c in conn]
                pj_params = [{'p_j':max(self.params['epsilon'], np.round(np.random.normal(self.params['p_j_d1'], self.params['std_p_j_d1']),1))} for c in conn]
                pij_params = [{'p_ij':max(self.params['positive_prior']*self.params['positive_prior']+self.params['positive_prior'], np.round(np.random.normal(self.params['p_ij_d1'], self.params['std_p_ij_d1']),1))} for c in conn]
                nest.SetStatus(conn, pi_params)
                nest.SetStatus(conn, pj_params)
                nest.SetStatus(conn, pij_params)
                # D2
                #nest.SetDefaults(self.params['dopa_bcpnn'], params=self.params['params_dopa_bcpnn_d2'])
                nest.DivergentConnect(self.states[nstates], self.strD2[nactions], model=self.params['synapse_d2'])
                conn = nest.GetConnections(source=self.states[nstates], target=self.strD2[nactions], synapse_model=self.params['synapse_d2'])
                delay_params = [{'delay':np.round(np.random.normal(self.params['delay_d2'], self.params['std_delay_d2']),1)} for c in conn]
                nest.SetStatus(conn, delay_params)
                pi_params = [{'p_i':max(self.params['epsilon'],np.round(np.random.normal(self.params['p_i_d2'], self.params['std_p_i_d2']),1))} for c in conn]
                pj_params = [{'p_j':max(self.params['epsilon'],np.round(np.random.normal(self.params['p_j_d2'], self.params['std_p_j_d2']),1))} for c in conn]
                pij_params = [{'p_ij':max(self.params['positive_prior']*self.params['positive_prior']+self.params['positive_prior'], np.round(np.random.normal(self.params['p_ij_d2'], self.params['std_p_ij_d2']),1))} for c in conn]
                nest.SetStatus(conn, pi_params)
                nest.SetStatus(conn, pj_params)
                nest.SetStatus(conn, pij_params)
            
        #connectiond1 = nest.GetConnections(target = [self.strD1[0][0]])
        #connectiond2 = nest.GetConnections(target = [self.strD2[0][0]])
        self.comm.barrier()
        if self.pc_id ==1:
            print 'GETCONNECT neuron 1'
            pp.pprint(nest.GetConnections(target= self.strD1[0]))
           # pp.pprint(nest.GetStatus(connectiond1))
           # print 'GETCONNECT neuron 2'
           # pp.pprint(nest.GetStatus(connectiond2))
        self.comm.barrier()
        

        # ####################
        # NOISE 
        # ##################### 
        self.noise_d1_exc = nest.Create('poisson_generator',1) 
        self.noise_d1_inh = nest.Create('poisson_generator',1) 
        self.noise_d2_exc = nest.Create('poisson_generator',1) 
        self.noise_d2_inh = nest.Create('poisson_generator',1) 
        self.noise_actions_exc = nest.Create('poisson_generator',1) 
        self.noise_actions_inh = nest.Create('poisson_generator',1) 
        self.noise_rp_exc = nest.Create('poisson_generator',1) 
        self.noise_rp_inh = nest.Create('poisson_generator',1) 

        for i in xrange(self.params['n_actions']):
            nest.DivergentConnect(self.noise_d1_exc, self.strD1[i], weight=self.params['noise_weight_d1_exc'], delay=self.params['noise_delay_d1_exc'])
            nest.DivergentConnect(self.noise_d1_inh, self.strD1[i], weight=self.params['noise_weight_d1_inh'], delay=self.params['noise_delay_d1_inh'])
            nest.DivergentConnect(self.noise_d2_exc, self.strD2[i], weight=self.params['noise_weight_d2_exc'], delay=self.params['noise_delay_d2_exc'])
            nest.DivergentConnect(self.noise_d2_inh, self.strD2[i], weight=self.params['noise_weight_d2_inh'], delay=self.params['noise_delay_d2_inh'])
            nest.DivergentConnect(self.noise_actions_exc, self.actions[i], weight=self.params['noise_weight_actions_exc'], delay=self.params['noise_delay_actions_exc'])
            nest.DivergentConnect(self.noise_actions_inh, self.actions[i], weight=self.params['noise_weight_actions_inh'], delay=self.params['noise_delay_actions_inh'])

        for i in xrange(self.params['n_states']*self.params['n_actions']):
            nest.DivergentConnect(self.noise_rp_exc, self.rp[i], weight = self.params['noise_weight_rp_exc'], delay=self.params['noise_delay_rp_exc'])
            nest.DivergentConnect(self.noise_rp_inh, self.rp[i], weight = self.params['noise_weight_rp_inh'], delay=self.params['noise_delay_rp_inh'])


            ## ????????????????????? add noise and variability to RP, Actions and Brainstem

        self.first_action_gid = np.min(self.actions[0]) - 1
        self.last_action_gid = np.max(self.actions[self.params['n_actions']-1])

    # #####################
 	# GETCONNECTIONS
	# #####################

	self.conn_dopa1 = nest.GetConnections(self.states[0], self.strD1[self.who], synapse_model='bcpnn_dopamine_synapse_d1')
	self.conn_dopa2 = nest.GetConnections(self.states[0], self.strD2[self.who], synapse_model='bcpnn_dopamine_synapse_d2')
	self.conn_d1 = []
	self.conn_d2 = []
	self.conn_habit0 = []
	self.conn_habit1 = []
	self.conn_habit2 = []
	self.conn_rp = []
	for i in xrange(self.params['n_actions']):
	    self.conn_d1.append(nest.GetConnections(self.states[self.who], self.strD1[i], synapse_model='bcpnn_dopamine_synapse_d1' ))
	    self.conn_d2.append(nest.GetConnections(self.states[self.who], self.strD2[i], synapse_model='bcpnn_dopamine_synapse_d2' ))
	    self.conn_habit0.append(nest.GetConnections(self.states[0], self.brainstem[i], synapse_model='bcpnn_synapse' ))
	    self.conn_habit1.append(nest.GetConnections(self.states[1], self.brainstem[i], synapse_model='bcpnn_synapse' ))
	    self.conn_habit2.append(nest.GetConnections(self.states[2], self.brainstem[i], synapse_model='bcpnn_synapse' ))
	for j in xrange(self.params['n_actions']*self.params['n_states']):
	    self.conn_rp.append( nest.GetConnections(self.rp[j], self.rew, synapse_model='bcpnn_dopa_synapse_RP')  )

        
        print "BG model completed"

    def create_brainstem(self):
        """
        Creates a new output population (brainstem)  and a static connection between actions and output. 
        """
        self.brainstem = {}
        self.recorder_brainstem = {}
        self.noise_bs_exc = nest.Create('poisson_generator',1) 
        self.noise_bs_inh = nest.Create('poisson_generator',1) 
        for i in xrange(self.params['n_actions']):
            self.brainstem[i] = nest.Create( self.params['model_brainstem_neuron'], self.params['num_brainstem_neurons'], params= self.params['param_brainstem_neuron'] )
            self.recorder_brainstem[i] = nest.Create("spike_detector", params= self.params['spike_detector_brainstem'])
            nest.SetStatus(self.recorder_brainstem[i],[{"to_file": True, "withtime": True, 'label' : self.params['brainstem_spikes_fn'] + str(i)}])
            nest.ConvergentConnect(self.brainstem[i], self.recorder_brainstem[i])
            nest.DivergentConnect(self.noise_bs_exc, self.brainstem[i], weight=self.params['noise_weight_bs_exc'], delay=self.params['noise_delay_bs_exc'])
            nest.DivergentConnect(self.noise_bs_inh, self.brainstem[i], weight=self.params['noise_weight_bs_inh'], delay=self.params['noise_delay_bs_inh'])
 
        print "Brainstem output created"

    def connect_brainstem(self):
        for i in xrange(self.params['n_actions']):
            for neur in self.brainstem[i]:
                nest.ConvergentConnect(self.actions[i], [neur], weight=self.params['weight_actions_brainstem'], delay=self.params['delay_actions_brainstem'])

    def connect_bcpnn_sensorimotor(self):
        """
        Creates a plastic connection from state populations to this output population
        """
        nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_states_brainstem'])
        for ns in xrange(self.params['n_states']):
            for na in xrange(self.params['n_actions']):
                nest.DivergentConnect(self.states[ns], self.brainstem[na], model=self.params['synapse_states_brainstem'] )

        print "Sensorimotor connection completed"

    def set_init(self):
        for i in xrange(self.params['n_states']):
            nest.SetStatus(self.input_poisson[i], {'rate': self.params['initial_poisson_input_rate']})
       # for j in xrange(self.params['n_actions']):
        nest.SetStatus(self.noise_d1_exc, {'rate': self.params['initial_noise_striatum_rate']})
        nest.SetStatus(self.noise_d2_exc, {'rate': self.params['initial_noise_striatum_rate']})

    def set_efference_copy(self, action):
        """
        Activates poisson generator to activate the selected action accordingly in the different pathways accordingly to the complementary activity.
        """
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.efference_copy[nactions], {'rate' : self.params['inactive_efference_rate']})
        #    print 'debug: EFFERENCE OFF for ACTION', nactions  , 'activity is: ',nest.GetStatus(self.efference_copy[nactions])[0]['rate'] 
        nest.SetStatus(self.efference_copy[action], {'rate' : self.params['active_full_efference_rate']})
        #print 'debug: EFFERENCE SET for ACTION',action ,' activity is: ',nest.GetStatus(self.efference_copy[action])[0]['rate'] 
        
    def stop_efference(self):
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.efference_copy[nactions], {'rate' : self.params['inactive_efference_rate']})
            #print 'debug: EFFERENCE OFF, activity is: ',nest.GetStatus(self.efference_copy[nactions])[0]['rate'] 
    
    
    
    def set_rp(self, state, action, gain, kappa):
        if self.params['params_dopa_bcpnn_RP']['dopamine_modulated']:
            nest.SetStatus(nest.GetConnections( self.rp[ action + state*self.params['n_actions'] ], self.rew ),{'gain':gain*self.params['gain_rp']})
        else:
            nest.SetStatus(nest.GetConnections( self.rp[ action + state*self.params['n_actions'] ], self.rew ),{'gain':gain, 'K':kappa})


    def trigger_reduce_pop_dopa(self, value):
        conn = nest.GetConnections(target=self.vt_dopa)
        value = value*self.params['num_rew_neurons']/100.
        parkinsonian = []
        self.comm.barrier()
        if self.pc_id == 0:
            parkinsonian = np.random.randint(conn[0][0], conn[-1][0], value)
        parkinsonian = self.comm.bcast(parkinsonian, root=0)
        self.comm.barrier()
        
        #print 'PARKINSON1 ', parkinsonian
        print 'PARKINSON2 ', conn
        for i in parkinsonian:
            for j in xrange(len(conn)):
                if conn[j][0] == i:
                    #nest.SetStatus([conn[j]], {'weight':0.})
                    nest.SetStatus([conn[j][0]], {'frozen':True})
                    print 'LOST ', conn[j], 'data ', nest.GetStatus([conn[j]])




    def set_state(self, state):
        """
        Informs BG about the current state. Used only when input state is internal to BG. Poisson population stimulated.
        """
        
        for i in range(self.params['n_states']):
            nest.SetStatus(self.input_poisson[i], {'rate' : self.params['inactive_poisson_input_rate']})
            #print 'debug: STATE OFF, activity is: ',nest.GetStatus(self.input_poisson[i])[0]['rate'] 
        nest.SetStatus(self.input_poisson[state], {'rate' : self.params['active_poisson_input_rate']})
        #print 'debug: STATE ON, activity is: ',nest.GetStatus(self.input_poisson[state])[0]['rate'] 

    def stop_state(self):
        """
        Stops poisson input to BG, no current state.
        """
        for i in range(self.params['n_states']):
            nest.SetStatus(self.input_poisson[i], {'rate' : self.params['inactive_poisson_input_rate']})
            #print 'debug: STATE OFF, activity is: ',nest.GetStatus(self.input_poisson[i])[0]['rate'] 

    def set_rest(self):
        """
        Informs BG about the current state. Used only when input state is internal to BG. Poisson population stimulated.
        """
        self.stop_state()
        self.baseline_reward()
        self.stop_efference()

    def get_action(self):
        """
        Returns the selected action. Calls a selection function e.g. softmax, hardmax, ...
        """
        new_event_gids = np.array([])
        for i_, recorder in enumerate(self.recorder_output.values()):
            all_events = nest.GetStatus(recorder)[0]['events']
            recent_event_idx = all_events['times'] > self.t_current
            if recent_event_idx.size > 0:
                new_event_gids = np.r_[new_event_gids, all_events['senders'][recent_event_idx]]
        if self.comm != None:
            gids_spiked, nspikes = utils.communicate_local_spikes(new_event_gids, self.comm)
        else:
            gids_spiked = new_event_gids.unique() - 1
            nspikes = np.zeros(len(new_event_gids))
            for i_, gid in enumerate(new_event_gids):
                nspikes[i_] = (new_event_gids == gid).nonzero()[0].size
        if sum(nspikes)==0:
            print '*******no spikes*******'
            winning_action = utils.communicate_action(self.comm, self.params['n_actions'])
        else:    
            #print 'gids_spiked ', gids_spiked
            #print 'nspikes ', nspikes
            all_actions_gids = np.arange(self.first_action_gid, self.last_action_gid)
            all_spikes = np.zeros(self.last_action_gid-self.first_action_gid)
            for gid in gids_spiked:
                all_spikes[all_actions_gids==gid] = nspikes[gids_spiked==gid]
        #    results = np.histogram(gids_spiked, bins=self.params['n_actions'], weights = nspikes)
        #    print 'results_histo_1 ', results
            results = np.histogram(all_actions_gids, bins=self.params['n_actions'], weights = all_spikes)
            #print 'results_histo_2 ', results
           # winning_nspikes = np.argmax(nspikes)
            randm = 0.
            winning_action = 0
            if self.params['softmax']:
                if self.comm.rank ==0:
                    randm = np.random.random()
                randm = self.comm.bcast(randm, root=0)
                self.comm.barrier()
                softmax = results[0]
                #print 'softmax_0', softmax
                softmax = softmax / np.sum(softmax)
                #print 'softmax_1', softmax
                softmax= 1. - softmax   #we want to select the action coded by the least active GPi/SNr population
                softmax = np.exp(self.params['temperature']*softmax) 
                softmax = softmax / np.sum(softmax)
                #print 'softmax_2', softmax
                for i in xrange(1,self.params['n_actions']):
                    softmax[i] += softmax[i-1]
                    if randm >= softmax[i-1]:
                        winning_action = int(i)
            else:
                #winning_gid = gids_spiked[winning_nspikes]
                #print 'winning gid: ', winning_gid
                #winning_action = self.recorder_output_gidkey[winning_gid+1]
                winning_action = np.argmin(results[0])
        print 'BG says (it %d, pc_id %d): do action %d' % (self.t_current / self.params['t_iteration'], self.pc_id, winning_action)
        self.t_current += self.params['t_iteration']
        return (winning_action)

    def set_reward(self, rew):
    # absolute value of the reward
        if rew:
            nest.SetStatus(self.poisson_rew, {'rate' : self.params['active_poisson_rew_rate']})
            #print 'debug: REWARD SET, activity is: ',nest.GetStatus(self.poisson_rew)[0]['rate'] 

    def baseline_reward(self):
        nest.SetStatus(self.poisson_rew, {'rate' : self.params['baseline_poisson_rew_rate']})
       # print 'debug: REWARD BASELINE, activity is: ', nest.GetStatus(self.poisson_rew)[0]['rate']
       # print 'KKAAPP D1' , nest.GetStatus(nest.GetConnections(self.states[0],self.strD1[0]))[0]
       # print 'KKAAPP D2' , nest.GetStatus(nest.GetConnections(self.states[0],self.strD2[0]))[0]


    def no_reward(self):
        nest.SetStatus(self.poisson_rew, {'rate' : self.params['inactive_poisson_rew_rate']})
        #print 'debug: REWARD OFF, activity is: ', nest.GetStatus(self.poisson_rew)[0]['rate']

    def set_noise(self):
        nest.SetStatus(self.noise_d1_exc, {'rate': self.params['noise_rate_d1_exc']}) 
        nest.SetStatus(self.noise_d1_inh, {'rate': self.params['noise_rate_d1_inh']}) 
        nest.SetStatus(self.noise_d2_exc, {'rate': self.params['noise_rate_d2_exc']}) 
        nest.SetStatus(self.noise_d2_inh, {'rate': self.params['noise_rate_d2_inh']}) 
        nest.SetStatus(self.noise_actions_exc, {'rate': self.params['noise_rate_actions_exc']}) 
        nest.SetStatus(self.noise_actions_inh, {'rate': self.params['noise_rate_actions_inh']}) 
        nest.SetStatus(self.noise_rp_exc, {'rate': self.params['noise_rate_rp_exc']}) 
        nest.SetStatus(self.noise_rp_inh, {'rate': self.params['noise_rate_rp_inh']}) 
        nest.SetStatus(self.noise_bs_exc, {'rate': self.params['noise_rate_bs_exc']}) 
        nest.SetStatus(self.noise_bs_inh, {'rate': self.params['noise_rate_bs_inh']}) 



    def set_weights(self, src_pop, tgt_pop, conn_mat_ee, src_pop_idx, tgt_pop_idx):
       # set the connection weight after having loaded the conn_mat_ee
       nest.SetStatus(nest.GetConnections(src_pop, tgt_pop), {'weight': conn_mat_ee[src_pop_idx, tgt_pop_idx]})
       # nest.SetStatus(nest.FindConnections(src_pop, tgt_pop), {'weight': conn_mat_ee[src_pop_idx, tgt_pop_idx]})
    
    def set_striosomes(self, state, action, weight):
        nest.SetStatus(nest.GetConnections(self.efference_copy[action], self.rp[action+ state*self.params['n_actions']]), {'weight':weight})


    def set_gain(self, gain):
        # implement option to change locally to d1 or d2 or RP

       for nstate in range(self.params['n_states']):
           for naction in range(self.params['n_actions']):
               # pp.pprint(nest.GetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction], self.params['synapse_d1'])))
              # nest.SetStatus(self.strD1[naction], {'gain':gain*self.params['gain_neuron']})
              # nest.SetStatus(self.strD2[naction], {'gain':gain*self.params['gain_neuron']})
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction], self.params['synapse_d1']), {'gain':gain*self.params['gain_d1']})
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD2[naction]), {'gain':gain*self.params['gain_d2']})
    	
     #  for index_rp in range(self.params['n_actions'] * self.params['n_states']):
     #       for naction in range(self.params['n_actions']):
     #          nest.SetStatus(nest.GetConnections(self.actions[naction], self.rp[index_rp % self.params['n_states']]), {'gain':gain})
     #       for nstate in range(self.params['n_states']):
     #          nest.SetStatus(nest.GetConnections(self.states[nstate], self.rp[int(index_rp / self.params['n_actions'])]), {'gain':gain})
     #       nest.SetStatus( self.rp[index_rp], {'gain':gain*self.params['gain_neuron']})
        
#       for nstates in range(self.params['n_states']):
#           #            print 'getstatus ' , nest.GetStatus(nest.FindConnections(self.states[nstates]))
#           nest.SetStatus([nest.FindConnections(self.states[nstates])], {'gain':gain})
#
#       for index_rp in range(self.params['n_actions']) :
#           nest.SetStatus([nest.FindConnections(self.actions[nactions])], {'gain':gain})
    
    def set_gain_dopa(self, gain):
        # implement option to change locally to d1 or d2 or RP

       for nstate in range(self.params['n_states']):
           for naction in range(self.params['n_actions']):
               # pp.pprint(nest.GetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction], self.params['synapse_d1'])))
              # nest.SetStatus(self.strD1[naction], {'gain':gain*self.params['gain_neuron']})
              # nest.SetStatus(self.strD2[naction], {'gain':gain*self.params['gain_neuron']})
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction], self.params['synapse_d1']), {'gain_dopa':gain*self.params['gain_dopa']})
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD2[naction]), {'gain_dopa':gain*self.params['gain_dopa']})

    def set_kappa_ON(self, k, state, action):
        # implement option to change locally to d1 or d2 or RP
        #To implement the opposite effect on the D1 and D2 MSNs of the dopamine release, -k is sent to D2

        if k< 0:
            nest.SetStatus( nest.GetConnections( self.states[state], self.strD2[action] ), {'k': -k} )
            nest.SetStatus( self.strD2[action], {'kappa': -k} ) 
        else:
            nest.SetStatus( nest.GetConnections( self.states[state], self.strD1[action] ), {'k': k} )
            nest.SetStatus( self.strD1[action], {'kappa': k} ) 
        
        if k<0.:
            conn = nest.GetConnections(source = self.states[state], target = self.rp[action+state*self.params['n_actions']], synapse_model = 'bcpnn_synapse')
            for c in nest.GetStatus(conn):
                if c['p_j'] > self.params['threshold']:
                    nest.SetStatus(nest.GetConnections(self.states[state], self.rp[action+state*self.params['n_actions']]), {'k':k})
                else:
                    print 'lOW p_j', c['p_j'] 

            conn = nest.GetConnections(source = self.actions[action], target = self.rp[action+state*self.params['n_actions']], synapse_model = 'bcpnn_synapse')
            for c in nest.GetStatus(conn):
                if c['p_j'] > self.params['threshold']:
                    nest.SetStatus(nest.GetConnections(self.actions[action], self.rp[action+state*self.params['n_actions']]), {'k':k})
                else:
                    print 'lOW p_j', c['p_j'] 

            nest.SetStatus(self.rp[state+action*self.params['n_states']], {'kappa':k} )
        else: 
            nest.SetStatus(nest.GetConnections(self.states[state], self.rp[action+state*self.params['n_actions']]), {'k':k})
            nest.SetStatus(nest.GetConnections(self.actions[action], self.rp[action+state*self.params['n_actions']]), {'k':k})
            nest.SetStatus(self.rp[state+action*self.params['n_states']], {'kappa':k} )




    def set_kappa_OFF(self):
        # implement option to change locally to d1 or d2 or RP
        #To implement the opposite effect on the D1 and D2 MSNs of the dopamine release, -k is sent to D2
       for nstate in range(self.params['n_states']):
           for naction in range(self.params['n_actions']):
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction]), {'k':0.})
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD2[naction]), {'k':0.})
       for nact in xrange(self.params['n_actions']):
           nest.SetStatus(self.strD1[nact], {'kappa':0.} )
           nest.SetStatus(self.strD2[nact], {'kappa':0.} )
       #nest.SetStatus(nest.GetConnections(self.states[state], self.strD1[action]), {'K': 0.} )
       #nest.SetStatus(self.strD1[action], {'kappa': 0.})



       for index_rp in range(self.params['n_actions'] * self.params['n_states']):
            for naction in range(self.params['n_actions']):
               nest.SetStatus(nest.GetConnections(self.actions[naction], self.rp[index_rp % self.params['n_states']]), {'k':0.})
            for nstate in range(self.params['n_states']):
    			nest.SetStatus(nest.GetConnections(self.states[nstate], self.rp[int(index_rp / self.params['n_actions'])]), {'k':0.})
            nest.SetStatus(self.rp[index_rp], {'kappa':0.} )




    def load_weights(self, training_params):
        """
        Connects the sensor layer (motion-prediction network, MPN) to the Basal Ganglia 
        based on the weights found in conn_folder
        """
        print 'debug', os.path.exists(training_params['d1_weights_fn'])
        print 'debug', training_params['d1_weights_fn']
        if not os.path.exists(training_params['d1_weights_fn']):
            # merge the connection files
            merge_pattern = training_params['d1_conn_fn_base']
            fn_out = training_params['d1_merged_conn_fn']
            utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)
      
        print 'Loading BG D1 connections from:', training_params['d1_merged_conn_fn']
        d1_conn_list = np.loadtxt(training_params['d1_merged_conn_fn'])



    def get_weights(self):
        """
        After training get the weights between the MPN state layer and the BG action layer
        """

        print 'Writing weights to files...'
        D1_conns = ''
        D2_conns = ''
        RP_conns = ''   #write code for the RP connections 
        for nactions in range(self.params['n_actions']):
            print 'action %d' % nactions

            conns = nest.GetConnections(self.states, self.strD1[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        D1_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w)

            conns = nest.GetConnections(self.states, self.strD2[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        D2_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w)

        fn_out = self.params['d1_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D1_f = file(fn_out, 'w')
        D1_f.write(D1_conns)
        D1_f.close()

        fn_out = self.params['d2_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D2_f = file(fn_out, 'w')
        D2_f.write(D2_conns)
        D2_f.close()
