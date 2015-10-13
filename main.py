import sys
import os
import nest
import BasalGanglia
import Reward
import json
import simulation_parameters
import utils
import numpy as np
import time
import tempfile
import mynest
import mynest_light
import pprint as pp


os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import pylab as pl


print 'INIT'
# Load BCPNN synapse and bias-iaf neuron module                                                                                                
if (not 'bcpnn_dopamine_synapse' in nest.Models()):
    nest.sr('(/cfs/milner/scratch/b/berthet/modules/bcpnndopa_module/share/ml_module/sli) addpath') #t/tully/sequences/share/nest/sli
    nest.Install('/cfs/milner/scratch/b/berthet/modules/bcpnndopa_module/lib/nest/ml_module') #t/tully/sequences/lib/nest/pt_module
   # nest.Install('ml_module')

try: 
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"
nest.ResetKernel()
nest.ResetKernel()
#USE_MPI = False
#pc_id, n_proc, comm = 0, 1, None
#print "MPI not used"
#nest.SetKernelStatus({'local_num_threads':12})
    

# load bcpnn synapse module and iaf neuron with bias
#if (not 'bcpnn_synapse' in nest.Models('synapses')):
#    nest.Install('pt_module')
nest.SetKernelStatus({"overwrite_files": True})
#nest.SetKernelStatus({"total_num_virtual_procs":20})
#N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]

################# RANDOM SEEDING

#msd = 44777333
#N_vp = comm.size
#pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
#nest.SetKernelStatus({'grng_seed': msd+N_vp})
#nest.SetKernelStatus({'rng_seeds':range(msd+N_vp+1, msd+2*N_vp+1 )})

#np.random.seed(np.random.randint(0,10000))


def save_spike_trains(params, iteration, stim_list, gid_list):
    n_units = len(stim_list)
    fn_base = params['input_st_fn_mpn']
    for i_ in xrange(n_units):
        if len(stim_list[i_]) > 0:
            fn = fn_base + '%d_%d.dat' % (iteration, gid_list[i_] - 1)
            np.savetxt(fn, stim_list[i_])


def remove_files_from_folder(folder):
    print 'Removing all files from folder:', folder
    path =  os.path.abspath(folder)
    cmd = 'rm  %s/*' % path
    print cmd
    os.system(cmd)


if __name__ == '__main__':


    print 'number of arguments ', len (sys.argv)

    if len(sys.argv) != 4: # program name + number of args
        sys.exit('Usage: %s' %sys.argv[0])
    args = map(float, sys.argv[1:])
    counter = int(args[1])

   #     param_fn = sys.argv[1]
   #     if os.path.isdir(param_fn): # go to the path containing the json object storing old parameters
   #         param_fn += '/Parameters/simulation_parameters.json' # hard coded subpath in ParameterContainer
   #     assert os.path.exists(param_fn), 'ERROR: Can not find %s - please give an existing parameter filename or folder name to re-run a simulation' % (param_fn)
   #     f = file(param_fn, 'r')
   #     print 'Loading parameters from', param_fn
   #     params = json.load(f)
    
    GP = simulation_parameters.global_parameters(args)
    if comm != None:
        comm.barrier()
    if pc_id==0:
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
    if comm != None:
        comm.barrier()
    params = GP.params

    nest.SetKernelStatus({"resolution":params['dt'] })
    t0 = time.time()
    # #####################################
    # TEMP STORAGE
    # #####################################

    #weights
    staten2D1 = np.empty(( params['n_recordings'], params['n_actions'] ))
    staten2D2 = np.empty(( params['n_recordings'], params['n_actions']))
    stateb2D1 = np.empty(( params['n_recordings'], params['n_actions'] ))
    stateb2D2 = np.empty(( params['n_recordings'], params['n_actions']))
    state2D1s = np.empty(( params['n_recordings'], params['n_states'] ))
    state2D2s = np.empty(( params['n_recordings'], params['n_states'] ))
    br_w0 = np.empty(( params['n_recordings'], params['n_actions']))
    br_w1 = np.empty(( params['n_recordings'], params['n_actions']))
    br_w2 = np.empty(( params['n_recordings'], params['n_actions']))
    rp_w = np.zeros(( params['n_recordings'], params['n_states'] * params['n_actions']))
    rp_bias = np.zeros(( params['n_recordings'], params['n_states'] * params['n_actions']))
    rp_pi = np.zeros(( params['n_recordings'], params['n_states'] * params['n_actions']))
    rp_pij = np.zeros(( params['n_recordings'], params['n_states'] * params['n_actions']))
    
    list_d1 = [ staten2D1, [],[],[],[],[],[],[], stateb2D1 ]
    list_d2 = [ staten2D2, [],[],[],[],[],[],[], stateb2D2 ]
    list_br = [ [],[],[],[], br_w0, br_w1, br_w2 ]
    list_rp = [ rp_w , rp_pi,rp_pij, rp_bias]
    
    
    if not params['light_record']:

        # e_ij_c
        state_D1_eijc = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        state_D2_eijc = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        # e_ij
        state_D1_eij = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        state_D2_eij = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        # p_i
        state_D1_pi = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        state_D2_pi = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        # p_j
        state_D1_pj = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        state_D2_pj = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        # p_ij
        state_D1_pij = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        state_D2_pij = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
        # e_ij_c
        state_D1_eijc = np.empty(params['n_actions'])
        state_D2_eijc = np. empty(params['n_actions'])
        # e_ij
        state_D1_eij = np.empty(params['n_actions'])
        state_D2_eij = np.empty(params['n_actions'])
        # p_i
        state_D1_pi =  np.empty(params['n_actions'])
        state_D2_pi =  np.empty(params['n_actions'])
        # p_j
        state_D1_pj =  np.empty(params['n_actions'])
        state_D2_pj =  np.empty(params['n_actions'])
        # p_ij
        state_D1_pij = np.empty(params['n_actions'])
        state_D2_pij = np.empty(params['n_actions'])


        action2D1 = np.empty(params['n_states'])
        action2D2 = np.empty(params['n_states'])
        stdaction2D1 = np.empty(params['n_states'])
        stdaction2D2 = np.empty(params['n_states'])
        br_pi = np.empty(params['n_actions'])
        br_pj = np.empty(params['n_actions'])
        br_pij = np.empty(params['n_actions'])
        br_eij = np.empty(params['n_actions'])

        #rp_pi = np.zeros(params['n_states'] * params['n_actions'])
        rp_pj = np.zeros(params['n_states'] * params['n_actions'])
        #rp_pij = np.zeros(params['n_states'] * params['n_actions'])



        list_d1 = [ staten2D1, action2D1, state_D1_pi, state_D1_pj, state_D1_pij, state_D1_eij, state_D1_eijc, stdaction2D1, state2D1s]
        list_d2 = [ staten2D2, action2D2, state_D2_pi, state_D2_pj, state_D2_pij, state_D2_eij, state_D2_eijc, stdaction2D2, state2D2s]
        list_br = [br_eij, br_pi, br_pj, br_pij, br_w0, br_w1, br_w2]
        #list_rp = [rp_w, rp_pi, rp_pj, rp_pij]
        list_rp = [rp_w, [], rp_pj, []]

    #kf = filtered k
    dopa_kf = np.array([])
    #k = gain_dopa * m
    dopa_k = np.array([])
    # m
    dopa_m = np.array([])
    # n
    dopa_n = np.array([])
    BG = BasalGanglia.BasalGanglia(params, comm)
    R = Reward.Reward(params)

    actions = np.empty(params['n_iterations']) 
    states  = np.empty(params['n_iterations']) 
    rewards = np.empty(params['n_iterations']) 
#    network_states_net= np.zeros((params['n_iterations'], 4))
    block = 0
    BG.set_noise()
    BG.baseline_reward()
    #BG.set_reward(1)
    BG.set_gain(0.)
    BG.set_init()
    if comm != None:
        comm.barrier()
    if params['light_record']:
        dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest_light.Simulate(params['t_init'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp,comm)
    else:
        dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest.Simulate(params['t_init'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp,comm)

    #BG.baseline_reward()
    #BG.set_gain(params['gain'])
    BG.set_gain(1.)
    BG.set_noise()
    BG.stop_efference()
#####   ###################         ###############
#####   ###################   LOOP  ###############
#####   ###################         ###############

    for iteration in xrange(params['n_iterations']):
        
        print 'ITERATION', iteration
        state = iteration % params['n_states']

        if params['trigger1'] and block==params['block_trigger1']:
            BG.trigger_reduce_pop_dopa(params['value_trigg_dopa_death'])
            #state = (iteration-1) % params['n_states']
            #block = int ((iteration-1) / params['block_len'])
            #BG.trigger_habit(True)
            #BG.trigger_change_dopa_zero(params['value_trigg_bias'])
            #params['baseline_poisson_rew_rate']= params['new_value']
            params['trigger1'] = False

             
        if params['trigger2'] and block==params['block_trigger2']:
            BG.trigger_habit(False)
            params['trigger2'] = False

#        state = utils.communicate_state(comm, params['n_states']) 
        BG.set_state(state)
        BG.baseline_reward()
       # BG.set_kappa_OFF()
        if comm != None:
            comm.barrier()
        if params['light_record']:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest_light.Simulate(params['t_selection'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        else:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest.Simulate(params['t_selection'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        if comm != None:
            comm.barrier()
        
       # weights_sim[iteration] = BG.get_weights(BG.states[0], BG.strD1[:])
        states[iteration] = state
        if iteration==0:
            actions[iteration] = utils.communicate_action(comm, params['n_actions'])
            BG.t_current += params['t_iteration']
        else:
            actions[iteration] = BG.get_action() # BG returns the selected action
        
        BG.set_efference_copy(actions[iteration])
        #BG.stop_state()
        BG.set_gain(0.)
        if comm != None:
            comm.barrier()
        BG.set_rp(states[iteration], actions[iteration], 0., 0.)
        ###BG.set_striosomes( states[iteration], actions[iteration], params['weight_efference_rp'])
        if params['light_record']:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest_light.Simulate(params['t_efference'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        else:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest.Simulate(params['t_efference'], 
                params['resolution'], params['n_actions'], params['n_states'], 
                BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp. comm)
        if comm != None:
            comm.barrier()

        if params['delay']:
            BG.stop_efference()
            BG.set_rest()
            BG.set_gain(1.)

        if params['light_record']:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest_light.Simulate(params['t_delay'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        else:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest.Simulate(params['t_delay'], 
                params['resolution'], params['n_actions'], params['n_states'], 
                BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        if comm != None:
            comm.barrier()
        #if comm != None:
        #    comm.barrier()
        rew = utils.communicate_reward(comm, R , state, actions[iteration], block )
        
        rewards[iteration] = rew
        
       # if params['delay']:
       #     BG.set_striosomes( states[iteration], actions[iteration], 0.)
        
        print 'REWARD =', rew
       # BG.set_gain(0.)
        if rew == 1:
            # BG.set_kappa_ON(params['rpe'], states[iteration], actions[iteration])
            BG.set_reward(rew)
        else:
            # BG.set_kappa_ON(-params['rpe'], states[iteration], actions[iteration])
            BG.no_reward()
        #BG.set_reward(rew)
        BG.set_rp(states[iteration], actions[iteration], 1. , 1.)
     #   BG.stop_efference()

        if comm != None:
            comm.barrier()
        if params['light_record']:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest_light.Simulate(params['t_reward'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        else:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest.Simulate(params['t_reward'], params['resolution'], params['n_actions'], params['n_states'], BG,dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        if comm != None:
            comm.barrier()

        ###BG.set_rp(states[iteration], actions[iteration], 0., 0. )
        #BG.set_gain(params['gain'])
        if not(params['delay']):
            BG.set_gain(1.)
      #  BG.set_kappa_OFF()
        BG.set_rest()
        if comm != None:
            comm.barrier()
        
        if params['light_record']:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest_light.Simulate(params['t_rest'], params['resolution'], params['n_actions'], params['n_states'], BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        else:
            dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp = mynest.Simulate(params['t_rest'], params['resolution'], params['n_actions'], params['n_states'], 
                                                                    BG, dopa_kf, dopa_k, dopa_m, dopa_n, list_d1, list_d2, list_br, list_rp, comm)
        if comm != None:
            comm.barrier()   
        block = int (iteration / params['block_len'])
    
### ########################################
    # END of SIMULATION LOOP
### ########################################
    print 'NEST simulation done for thead ', pc_id, ' process ', counter
    if pc_id == 0:
        np.savetxt(params['weights_d1_multi_fn']+'_'+str(counter) , list_d1[0])
        np.savetxt(params['weights_d2_multi_fn']+'_'+str(counter) , list_d2[0])
        np.savetxt(params['weights_rp_multi_fn']+'_'+str(counter) , list_rp[0])
        np.savetxt(params['rewards_multi_fn']+'_'+str(counter) , rewards)


    if params['record_spikes']:
        if counter == 0:
            if pc_id == 0:
                binsize = params['binsize_histo_raster']
                np.savetxt(params['actions_taken_fn'], actions)
                np.savetxt(params['states_fn'], states)
                np.savetxt(params['rewards_fn'], rewards)
                exc_sptimes = nest.GetStatus(BG.recorder_d1[0])[0]['events']['times']
                for i_proc in xrange(1,n_proc ):
                    exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc, tag=0 )]
               
            else:
                comm.send(nest.GetStatus(BG.recorder_d1[0])[0]['events']['times'],dest=0, tag=0)
            if comm != None:
                comm.barrier()
            if pc_id == 0:
                exc_spids = nest.GetStatus(BG.recorder_d1[0])[0]['events']['senders']
                for i_proc in xrange(1, n_proc):
                    exc_spids = np.r_[exc_spids, comm.recv(source=i_proc, tag=1)]
                pl.figure(33)
                pl.subplot(211)

                pl.scatter(exc_sptimes, exc_spids,s=1.)
                pl.xlim([0,params['t_sim']])
                bins=np.arange(0, params['t_sim']+1, binsize)
                c_exc,b = np.histogram(exc_sptimes,bins=bins)
                rate_exc = c_exc*(1000./binsize)*(1./params['num_msn_d1'])
                pl.subplot(212)
                pl.plot(b[0:-1],rate_exc)
                pl.xlim([0,params['t_sim']])
                pl.title('firing rate of STR D1 action 0')
                pl.savefig('fig3_firingrate.pdf')
            else:
                comm.send(nest.GetStatus(BG.recorder_d1[0])[0]['events']['senders'],dest=0, tag=1)
            if comm != None:
                comm.barrier()
            
           # if comm != None:
           #     comm.barrier()
##################
            if pc_id == 0:
                exc_sptimes = nest.GetStatus(BG.recorder_rew)[0]['events']['times']
                for i_proc in xrange(1,n_proc ):
                    exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc, tag=2)]
            else:
                comm.send(nest.GetStatus(BG.recorder_rew)[0]['events']['times'],dest=0, tag=2)
            if comm != None:
                comm.barrier()
            if pc_id == 0:
                exc_spids = nest.GetStatus(BG.recorder_rew)[0]['events']['senders']
                for i_proc in xrange(1, n_proc):
                    exc_spids = np.r_[exc_spids, comm.recv(source=i_proc, tag = 3)]
                pl.figure(34)
                pl.subplot(211)
                pl.scatter(exc_sptimes, exc_spids,s=1.)
                pl.xlim([0,params['t_sim']])
                bins=np.arange(0, params['t_sim']+1, binsize)
                c_exc,b = np.histogram(exc_sptimes,bins=bins)
                rate_exc = c_exc*(1000./binsize)*(1./params['num_rew_neurons'])
                pl.subplot(212)
                pl.plot(b[0:-1],rate_exc)
                pl.xlim([0,params['t_sim']])
                pl.title('firing rate of dopa neurons')
                pl.savefig('fig7_dopafiringrate.pdf')
            else:
                comm.send(nest.GetStatus(BG.recorder_rew)[0]['events']['senders'],dest=0, tag=3)
            if comm != None:
                comm.barrier()
##########################
            if pc_id == 0:
                exc_sptimes = nest.GetStatus(BG.recorder_states[0])[0]['events']['times']
                for i_proc in xrange(1,n_proc ):
                    exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc, tag=4)]
            else:
                comm.send(nest.GetStatus(BG.recorder_states[0])[0]['events']['times'],dest=0, tag=4)
            if comm != None:
                comm.barrier()
            if pc_id == 0:
                exc_spids = nest.GetStatus(BG.recorder_states[0])[0]['events']['senders']
                for i_proc in xrange(1, n_proc):
                    exc_spids = np.r_[exc_spids, comm.recv(source=i_proc, tag = 5)]
                pl.figure(35)
                pl.subplot(211)
                pl.scatter(exc_sptimes, exc_spids,s=1.)
                pl.xlim([0,params['t_sim']])
                bins=np.arange(0, params['t_sim']+1, binsize)
                c_exc,b = np.histogram(exc_sptimes,bins=bins)
                rate_exc = c_exc*(1000./binsize)*(1./params['num_neuron_states'])
                pl.subplot(212)
                pl.plot(b[0:-1],rate_exc)
                pl.xlim([0,params['t_sim']])
                pl.title('firing rate of states neurons')
                pl.savefig('fig10_states_firingrate.pdf')
            else:
                comm.send(nest.GetStatus(BG.recorder_states[0])[0]['events']['senders'],dest=0, tag=5)
            if comm != None:
                comm.barrier()
#######################
            if pc_id == 0:
                exc_sptimes = nest.GetStatus(BG.recorder_gpi[0])[0]['events']['times']
                for i_proc in xrange(1,n_proc ):
                    exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc, tag=8)]
            else:
                comm.send(nest.GetStatus(BG.recorder_gpi[0])[0]['events']['times'],dest=0, tag=8)
            if comm != None:
                comm.barrier()
            if pc_id == 0:
                exc_spids = nest.GetStatus(BG.recorder_gpi[0])[0]['events']['senders']
                for i_proc in xrange(1, n_proc):
                    exc_spids = np.r_[exc_spids, comm.recv(source=i_proc, tag = 9)]
                pl.figure(315)
                pl.subplot(211)
                pl.scatter(exc_sptimes, exc_spids,s=1.)
                pl.xlim([0,params['t_sim']])
                bins=np.arange(0, params['t_sim']+1, binsize)
                c_exc,b = np.histogram(exc_sptimes,bins=bins)
                rate_exc = c_exc*(1000./binsize)*(1./params['num_gpi'])
                pl.subplot(212)
                pl.plot(b[0:-1],rate_exc)
                pl.xlim([0,params['t_sim']])
                pl.title('firing rate of GPi neurons')
                pl.savefig('fig12_gpi_firingrate.pdf')
            else:
                comm.send(nest.GetStatus(BG.recorder_gpi[0])[0]['events']['senders'],dest=0, tag=9)
            if comm != None:
                comm.barrier()
#######################
            if pc_id == 0:
                exc_sptimes = nest.GetStatus(BG.recorder_brainstem[0])[0]['events']['times']
                for i_proc in xrange(1,n_proc ):
                    exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc, tag=12)]
            else:
                comm.send(nest.GetStatus(BG.recorder_brainstem[0])[0]['events']['times'],dest=0, tag=12)
            if comm != None:
                comm.barrier()
            if pc_id == 0:
                exc_spids = nest.GetStatus(BG.recorder_brainstem[0])[0]['events']['senders']
                for i_proc in xrange(1, n_proc):
                    exc_spids = np.r_[exc_spids, comm.recv(source=i_proc, tag = 13)]
                pl.figure(31525)
                pl.subplot(211)
                #pl.scatter(exc_sptimes, exc_spids,s=1.)
                pl.plot(exc_sptimes)
                pl.plot(exc_spids)
                pl.xlim([0,params['t_sim']])
                bins=np.arange(0, params['t_sim']+1, binsize)
                c_exc,b = np.histogram(exc_sptimes,bins=bins)
                rate_exc = c_exc*(1000./binsize)*(1./params['num_brainstem_neurons'])
                pl.subplot(212)
                pl.plot(b[0:-1],rate_exc)
                pl.xlim([0,params['t_sim']])
                pl.title('firing rate of brainstem neurons')
                pl.savefig('fig13_brainstem_firingrate.pdf')
            else:
                comm.send(nest.GetStatus(BG.recorder_gpi[0])[0]['events']['senders'],dest=0, tag=13)
            if comm != None:
                comm.barrier()
#######################
            if pc_id == 0:
                exc_sptimes = nest.GetStatus(BG.recorder_d2[0])[0]['events']['times']
                for i_proc in xrange(1,n_proc ):
                    exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc, tag=6)]
            else:
                comm.send(nest.GetStatus(BG.recorder_d2[0])[0]['events']['times'],dest=0, tag=6)
            if comm != None:
                comm.barrier()
            if pc_id == 0:
                exc_spids = nest.GetStatus(BG.recorder_d2[0])[0]['events']['senders']
                for i_proc in xrange(1, n_proc):
                    exc_spids = np.r_[exc_spids, comm.recv(source=i_proc, tag = 7)]
                pl.figure(36)
                pl.subplot(211)
                pl.scatter(exc_sptimes, exc_spids,s=1.)
                pl.xlim([0,params['t_sim']])
                bins=np.arange(0, params['t_sim']+1, binsize)
                c_exc,b = np.histogram(exc_sptimes,bins=bins)
                rate_exc = c_exc*(1000./binsize)*(1./params['num_msn_d2'])
                pl.subplot(212)
                pl.plot(b[0:-1],rate_exc)
                pl.xlim([0,params['t_sim']])
                pl.title('firing rate of D2 neurons')
                pl.savefig('fig11_D2firingrate.pdf')
            else:
                comm.send(nest.GetStatus(BG.recorder_d2[0])[0]['events']['senders'],dest=0, tag=7)
            if comm != None:
                comm.barrier()
#######################
            if pc_id == 0:
                exc_sptimes = nest.GetStatus(BG.recorder_rp[0])[0]['events']['times']
                for i_proc in xrange(1,n_proc ):
                    exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc, tag=10)]
            else:
                comm.send(nest.GetStatus(BG.recorder_rp[0])[0]['events']['times'],dest=0, tag=10)
            if comm != None:
                comm.barrier()
            if pc_id == 0:
                exc_spids = nest.GetStatus(BG.recorder_rp[0])[0]['events']['senders']
                for i_proc in xrange(1, n_proc):
                    exc_spids = np.r_[exc_spids, comm.recv(source=i_proc, tag = 11)]
                pl.figure(316)
                pl.subplot(211)
                pl.scatter(exc_sptimes, exc_spids,s=1.)
                pl.xlim([0,params['t_sim']])
                bins=np.arange(0, params['t_sim']+1, binsize)
                c_exc,b = np.histogram(exc_sptimes,bins=bins)
                rate_exc = c_exc*(1000./binsize)*(1./params['num_rp_neurons'])
                pl.subplot(212)
                pl.plot(b[0:-1],rate_exc)
                pl.xlim([0,params['t_sim']])
                pl.title('firing rate of striosomes neurons')
                pl.savefig('fig14_rp_firingrate.pdf')
            else:
                comm.send(nest.GetStatus(BG.recorder_rp[0])[0]['events']['senders'],dest=0, tag=11)
            if comm != None:
                comm.barrier()
            #CC.get_weights(, BG)  implement in BG or utils
            print 'Im process ', pc_id, 'VTmod is ', nest.GetStatus(BG.vt_dopa) 

        # print 'weight simu ', weights_sim
    #if pc_id ==3:	
        #print 'bias_d1', list_d1[8]
        #print 'bias_d2', list_d2[8]
        #print 'bias_rp', list_rp[3]
       # pl.figure(301)
       # pl.subplot(211)
       # pl.title('D1')
       # pl.plot(list_d1[8])
       # pl.ylabel(r'$b_{j}$')
       # pl.subplot(212)
       # pl.title('D2')
       # pl.plot(list_d2[8])
       # pl.ylabel(r'$b_{j}$')
       # pl.xlabel('trials')
       # pl.suptitle('Computed biases from state '+str(params['recorded'])+' to all actions')
       # pl.savefig('fig301_bias_allactions.pdf')

       # pl.figure(302)
       # pl.title('RP bias')
       # pl.plot(list_rp[3])
       # pl.ylabel(r'$b_{j}$')
       # pl.xlabel('trials')
       # pl.savefig('fig302_bias_rp.pdf')

    if pc_id ==1:	
        t1 = time.time() - t0
        print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
        
        print 'Isnan', np.isnan(np.sum(list_d1[0]))

        print 'LIST d1' , list_d1
        print 'D1[1]', list_d1[1]

        print 'learning completed'
        pl.figure(1)
        pl.subplot(211)
        pl.title('D1')
        pl.plot(list_d1[0])
        pl.ylabel(r'$w_{0j}$')
        pl.subplot(212)
        pl.title('D2')
        pl.plot(list_d2[0])
        pl.ylabel(r'$w_{0j}$')
        pl.xlabel('trials')
        pl.suptitle('Computed weights from state '+str(params['recorded'])+' to all actions')
        pl.savefig('fig1_allactions.pdf')

        if not params['light_record']:

            pl.figure(2)
            pl.subplot(211)
            pl.plot(list_d1[1])
            pl.title('D1')
            pl.ylabel(r'$w_{0j}$')
            pl.subplot(212)
            pl.plot(list_d2[1])
            pl.ylabel(r'$w_{0j}$')
            pl.title('D2')
            pl.xlabel('trials')
            pl.suptitle('Computed weights from all states to action 0')
            pl.savefig('fig2_allstates.pdf')
    
    if not params['light_record']:
        if pc_id == 2:
            pl.figure(3)
            pl.subplot(221)
            pl.title('D1')
            pl.plot(list_d1[5], label='e_ij' )
            pl.ylabel(r'$e_{trace}$')
            pl.legend()
            pl.subplot(222)
            pl.title('D2')
            pl.plot(list_d2[5], label='e_ij' )
            pl.subplot(223)
            pl.title('D1')
            pl.plot(list_d1[6], label='e_ijc' )
            pl.xlabel('trials')
            pl.ylabel(r'$e_{trace}$')
            pl.legend()
            pl.subplot(224)
            pl.title('D2')
            pl.plot(list_d2[6], label='e_ijc' )
            pl.xlabel('trials')
            pl.legend()
            pl.suptitle('Computed e traces from state 0 to all actions')
            pl.savefig('fig5_etraces.pdf')
            
            pl.figure(4)
            pl.subplot(221)
            pl.title('D1')
            pl.plot(list_d1[2], label='p_i' )
            pl.plot(list_d1[3], label='p_j' )
            pl.ylabel(r'$p_{trace}$')
            pl.legend()
            pl.subplot(222)
            pl.title('D2')
            pl.plot(list_d2[2], label='p_i' )
            pl.plot(list_d2[3], label='p_j' )
            pl.subplot(223)
            pl.title('D1')
            pl.plot(list_d1[4], label='p_ij' )
            pl.ylabel(r'$p_{trace}$')
            pl.xlabel('trials')
            pl.legend()
            pl.subplot(224)
            pl.title('D2')
            pl.plot(list_d2[4], label='p_ij' )
            pl.xlabel('trials')
            pl.legend()
            pl.suptitle('Computed p traces from state 0 to all states')
            pl.savefig('fig4_ptraces.pdf')

    if pc_id == 2:
        pl.figure(5)
        pl.plot(dopa_kf, label='kf')
        #pl.plot(dopa_k, label='k')
        pl.plot(dopa_m, label = 'm')
        pl.plot(dopa_n, label = 'n')
        pl.title('dopamine RPE dynamics' )
        pl.xlabel('time in '+ str(params['resolution'])+ 'ms')
        pl.legend()
        pl.ylabel('variable value')
        pl.savefig('fig6_dopa.pdf')
        pl.figure(1005)
        #pl.plot(dopa_kf, label='kf')
        #pl.plot(dopa_k, label='k')
        pl.plot(dopa_m, label = 'm')
        pl.plot(dopa_n, label = 'n')
        pl.title('dopamine RPE dynamics' )
        pl.xlabel('time in '+ str(params['resolution'])+ 'ms')
        pl.legend()
        pl.ylabel('variable value')
        pl.savefig('fig60_dopa.pdf')

        if not params['light_record']:
            pl.figure(88)
            pl.plot(list_d1[7], label='w std D1')
            pl.plot(list_d2[7], label='w std D2')
            pl.title('standard deviation incoming weights to action 0')
            pl.xlabel('trials')
            pl.legend()
            pl.savefig('fig88_stdw.pdf')

            pl.figure(89)
            pl.subplot(211)
            #pl.plot(list_br[0], label='eij')
            pl.plot(list_br[1], label='pi')
            pl.plot(list_br[2], label='pj')
            
            pl.plot(list_br[3], label='pij')
            pl.subplot(212)
            pl.plot(list_br[4], label='w0')
            pl.plot(list_br[5], label='w1')
            pl.plot(list_br[6], label='w2')
            pl.title('Brainstem traces + weights from state 0')
            pl.xlabel('trials')
            pl.legend()
            pl.savefig('fig89_brainstem.pdf')
        else:
            pl.figure(89)
            pl.plot(list_br[4], label='w0')
            pl.plot(list_br[5], label='w1')
            pl.plot(list_br[6], label='w2')
            pl.title('Brainstem weights from state 0')
            pl.xlabel('trials')
            pl.legend()
            pl.savefig('fig89_brainstem.pdf')

   # if pc_id == 0:
   #     for i_proc in xrange(1,n_proc ):
   #         dopa_n  += comm.recv(source=i_proc, tag=6)
   #     dopa_m = dopa_n + params['params_dopa_bcpnn']['b']
   #     dopa_k = dopa_m * params['params_dopa_bcpnn']['gain_dopa'] 
   #     dopa_kf = np.power(dopa_k,params['params_dopa_bcpnn']['k_pow'])
   #     pl.figure(8)
   #     pl.plot(dopa_k, label='k')
   #     pl.plot(dopa_kf, label='kf')
   #     pl.plot(dopa_m, label = 'm')
   #     pl.plot(dopa_n, label = 'n')
   #     pl.title('dopa parameters dynamics' )
   #     pl.xlabel('trials')
   #     pl.legend()
   #     pl.ylabel('parameter value')
   #     pl.savefig('fig8_dopa_mpi.pdf')
   #     pl.figure(8878)
   #     pl.plot(dopa_m, label = 'm')
   #     pl.plot(dopa_n, label = 'n')
   #     pl.title('dopa parameters dynamics' )
   #     pl.xlabel('trials')
   #     pl.legend()
   #     pl.ylabel('parameter value')
   #     pl.savefig('fig889_dopa_mpi.pdf')
   # else:
   #     comm.send(dopa_n, dest=0, tag=6)

   # if pc_id != 0:
   #     pl.figure(1300+pc_id)
   #     pl.plot(dopa_k, label='k')
   #     pl.plot(dopa_kf, label='kf')
   #     pl.plot(dopa_m, label = 'm')
   #     pl.plot(dopa_n, label = 'n')
   #     pl.title('dopa parameters dynamics for process '+ str(pc_id) )
   #     pl.xlabel('trials')
   #     pl.legend()
   #     pl.ylabel('parameter value')
   #     pl.savefig('fig'+str(1300+pc_id)+'_dopa.pdf')
    if not params['light_record']:
        nd1= len(nest.GetConnections(BG.states[0], BG.strD1[BG.who]))
        nd2= len(nest.GetConnections(BG.states[0], BG.strD2[BG.who]))
        sample = int(params['t_sim'] / params['resolution'])    
        #a1 = np.empty((nd1,sample))
        a1 = np.zeros((nd1,len(list_d1[8][0])/nd1))
        b1 = np.zeros((nd1,len(list_d1[8][1])/nd1))
        c1 = np.zeros((nd1,len(list_d1[8][2])/nd1))
        #b1 = np.empty((nd1,sample))
        #c1 = np.empty((nd1,sample))
        
        a2 = np.zeros((nd2,len(list_d2[8][0])/nd2))
        b2 = np.zeros((nd2,len(list_d2[8][1])/nd2))
        c2 = np.zeros((nd2,len(list_d2[8][2])/nd2))
        #a2 = np.empty((nd2,sample))
        #b2 = np.empty((nd2,sample))
        #c2 = np.empty((nd2,sample))
        
        
        print 'sizetest a1 ', len(a1), 'nd1 ', nd1, 'sample ', sample
        print 'sizetest listd1 ', len(list_d1), len(list_d1[8]), len(list_d1[8][0])


        for i in xrange(nd1):
            for j in xrange(len(list_d1[8][0])/nd1 -nd1):
                a1[i,j]=list_d1[8][0][1+i+nd1*j]
        for i in xrange(nd1):
            for j in xrange(len(list_d1[8][1])/nd1 -nd1):
                b1[i,j]=list_d1[8][1][1+i+nd1*j]
        for i in xrange(nd1):
            for j in xrange(len(list_d1[8][2])/nd1 -nd1):
                c1[i,j]=list_d1[8][2][1+i+nd1*j]
        

        pl.figure(887)
        pl.plot(np.transpose(a1))
        pl.title('state 0 to D1['+str(BG.who)+']')
        pl.savefig('fig'+str(810+pc_id)+'_singleW_a1.pdf')

        pl.figure(888)
        pl.plot(np.transpose(b1))
        pl.title('state 1 to D1['+str(BG.who)+']')
        pl.savefig('fig'+str(850+pc_id)+'_singleW_b1.pdf')

        
        pl.figure(889)
        pl.plot(np.transpose(c1))
        pl.title('state 2 to D1['+str(BG.who)+']')
        pl.savefig('fig'+str(880+pc_id)+'_singleW_c1.pdf')
        
        
        
        for i in xrange(nd2):
            for j in xrange(len(list_d2[8][0])/nd2 -nd2):
                a2[i,j]=list_d2[8][0][1+i+nd2*j]
        for i in xrange(nd2):
            for j in xrange(len(list_d2[8][1])/nd2 -nd2):
                b2[i,j]=list_d2[8][1][1+i+nd2*j]
        for i in xrange(nd2):
            for j in xrange(len(list_d2[8][2])/nd2 -nd2):
                c2[i,j]=list_d2[8][2][1+i+nd2*j]
        pl.figure(897)
        pl.plot(np.transpose(a2))
        pl.title('state 0 to D2['+str(BG.who)+']')
        pl.savefig('fig'+str(810+pc_id)+'_singleW_a2.pdf')

        pl.figure(898)
        pl.plot(np.transpose(b2))
        pl.title('state 1 to D2['+str(BG.who)+']')
        pl.savefig('fig'+str(850+pc_id)+'_singleW_b2.pdf')

        
        pl.figure(899)
        pl.plot(np.transpose(c2))
        pl.title('state 2 to D2['+str(BG.who)+']')
        pl.savefig('fig'+str(880+pc_id)+'_singleW_c2.pdf')

    print 'DATA management completed for thread ', pc_id, ' process ', counter

    if pc_id == 1:

        # print 'list_rp_check_0'
        # pp.pprint(list_rp[0])
        # print 'list_rp_check_1'
        # pp.pprint(list_rp[1])
        # print 'list_rp_check_2'
        # pp.pprint(list_rp[2])
        # print 'list_rp_check_3'
        # pp.pprint(list_rp[3])

        pl.figure(9001)
        pl.plot(list_rp[0], label='Wij')
        pl.title('Weight RP/Rew')
        pl.savefig('fig9001_meanWijRP.pdf')
        
       # pl.figure(9033)
       # pl.plot(list_rp[1], label='Pi')
       # pl.title('Traces RP Pi')
       # pl.legend()
       # pl.savefig('fig_meantracesRPi.pdf')
       # pl.figure(9034)
       # pl.plot(list_rp[2], label='Pij')
       # pl.title('Traces RP Pij')
       # pl.legend()
       # pl.savefig('fig_meantracesRPij.pdf')
       # pl.figure(9035)
       # pl.plot(list_rp[3], label='Pj')
       # pl.title('Traces RP Pj')
       # pl.legend()
       # pl.savefig('fig_meantracesRPj.pdf')
        if not params['light_record']:
            pl.figure(9002)
            pl.plot(list_rp[1], label='Pi')
            pl.plot(list_rp[2], label='Pj')
            pl.plot(list_rp[3], label='Pij')
            pl.title('Traces RP/Rew')
            pl.legend()
            pl.savefig('fig9002_meantracesRP.pdf')
        
        print 'States ', states
        print 'Actions ', actions
        print 'Rewards ', rewards

    print 'Thread ', pc_id, ' from process ', counter, ' DONE'

    #print 'RANDOM for pc_id ', pc_id , ' is ', np.random.randint(100)
