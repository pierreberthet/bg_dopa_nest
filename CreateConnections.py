import nest
import numpy as np
import os
import utils

class CreateConnections(object):

    def __init__(self, params, comm=None):
        
        self.params = params

        nest.CopyModel('static_synapse', 'mpn_bg_exc', \
                {'weight': self.params['w_exc_mpn_bg'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])
        self.comm = comm
        if comm != None:
            self.pc_id = comm.rank
            self.n_proc = comm.size

    def connect_mt_to_bg(self, src_net, tgt_net):
        """
        The NEST simulation should run for some pre-fixed time
        Keyword arguments:
        src_net, tgt_net -- the source and the target network

        """

        for nactions in range(self.params['n_actions']):
            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d1_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD1[nactions], model=self.params['synapse_d1_MT_BG'])

            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d2_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD2[nactions], model=self.params['synapse_d2_MT_BG'])


    def connect_mt_to_bg_after_training(self, mpn_net, bg_net, training_params):
        """
        Connects the sensor layer (motion-prediction network, MPN) to the Basal Ganglia 
        based on the weights found in conn_folder
        """
        print 'debug', os.path.exists(training_params['mpn_bgd1_merged_conn_fn'])
        print 'debug', training_params['mpn_bgd1_merged_conn_fn']
        if not os.path.exists(training_params['mpn_bgd1_merged_conn_fn']):
            # merge the connection files
            merge_pattern = training_params['mpn_bgd1_conn_fn_base']
            fn_out = training_params['mpn_bgd1_merged_conn_fn']
            utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)
      
        print 'Loading MPN - BG D1 connections from:', training_params['mpn_bgd1_merged_conn_fn']
        mpn_d1_conn_list = np.loadtxt(training_params['mpn_bgd1_merged_conn_fn'])



    def get_weights(self, src_pop, tgt_pop):
        """
        After training get the weights between the MPN state layer and the BG action layer
        """

        print 'Writing weights to files...'
        D1_conns = ''
        D2_conns = ''
        for nactions in range(self.params['n_actions']):
            print 'action %d' % nactions

            conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD1[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        D1_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w)

            conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD2[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        D2_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w)

        fn_out = self.params['mpn_bgd1_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D1_f = file(fn_out, 'w')
        D1_f.write(D1_conns)
        D1_f.close()

        fn_out = self.params['mpn_bgd2_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D2_f = file(fn_out, 'w')
        D2_f.write(D2_conns)
        D2_f.close() 

    def get_connection_kernel(self, src_gid):
        return 0
