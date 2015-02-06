import nest
import numpy as np
import pprint as pp

def Simulate(time, resolution, nactions, nstates, BG, dopakf, dopak, dopam, dopan, list_d1, list_d2, list_br, list_rp, comm):
    # /////////// REMOVE conn is a list, conn[0] = D1, conn[1] = D2, conn[2] = sensorimotor, conn[3] will be RP(to be defined),
    # dopa are array storing dopamine VT value.
    # list_* are lists storing the traces and weights of specific pathways
    #       list_d1[0] = 
    #       list_d1[1] = 
    #       list_d1[2] = 
    #       list_d1[3] = 
    #       list_d1[4] = 
    #       list_d1[5] = 
    #       list_d1[6] = 

    size = comm.Get_size()
    rank = comm.Get_rank()

    sim_time = time / resolution 
    dkf = np.empty(sim_time)
    dk = np.empty(sim_time)
    dm = np.empty(sim_time)
    dn = np.empty(sim_time)
#    sd1w = np.empty(nactions)
#    sd2w = np.empty(nactions)
#    sd1sw = np.empty((nstates, len(conn_dopa)))
#    sd2sw = np.empty((nstates, len(conn_dopa2)))
#    brw0 = np.empty(nactions)
#    brw1 = np.empty(nactions)
#    brw2 = np.empty(nactions)
    #rp_w = np.empty(nactions*nstates)
    
    for i in xrange(int(sim_time)):
        #        print 'SIMULATE ', resolution, 'number ', i, 'out of ', int(sim_time), 'BG_count:', BG.rec_count 
        nest.Simulate(resolution)
        dkf[i], dk[i], dm[i], dn[i] =  nest.GetStatus(BG.conn_dopa2, ['k_filtered', 'k', 'm', 'n'])[0]
        for q in xrange(nactions):
            #conn = nest.GetConnections(source = BG.states[BG.who], target = BG.strD1[q], synapse_model = 'bcpnn_dopamine_synapse_d1')
            #list_d1[0][BG.rec_count,q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(BG.conn_d1[q])])
            #list_d1[8][BG.rec_count,q] = np.mean([np.log(a['p_j']) for a in nest.GetStatus(BG.conn_d1[q])])
            #sd1w[q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            #conn = nest.GetConnections(source = BG.states[BG.who], target = BG.strD2[q], synapse_model = 'bcpnn_dopamine_synapse_d2')
            #sd2w[q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            #list_d2[0][BG.rec_count,q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(BG.conn_d2[q])])
            #list_d2[8][BG.rec_count,q] = np.mean([np.log(a['p_j']) for a in nest.GetStatus(BG.conn_d2[q])])
            #conn = nest.GetConnections(source = BG.states[BG.who], target = BG.brainstem[q], synapse_model = 'bcpnn_synapse')
            list_br[4][BG.rec_count,q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(BG.conn_habit0[q])])
            #conn = nest.GetConnections(source = BG.states[1], target = BG.brainstem[q], synapse_model = 'bcpnn_synapse')
            list_br[5][BG.rec_count,q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(BG.conn_habit1[q])])
            #conn = nest.GetConnections(source = BG.states[2], target = BG.brainstem[q], synapse_model = 'bcpnn_synapse')
            list_br[6][BG.rec_count,q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(BG.conn_habit2[q])])

            if rank == 0:
                #conn_d1 = (nest.GetConnections( source = BG.states[BG.who], target=BG.strD1[q], synapse_model='bcpnn_dopamine_synapse_d1' ))
                log = [np.log(a['p_ij']/(a['p_i']*a['p_j'])) for a in nest.GetStatus(BG.conn_d1[q]) ]
                #print 'lolog1', log, 'len = ', len(log), ' avg = ', np.mean(log)
                for i_proc in xrange(1,size):
                    log = np.r_[log, comm.recv(source=i_proc)]
            else:
                data = [np.log(a['p_ij']/ (a['p_j']* a['p_i'])) for a in nest.GetStatus(BG.conn_d1[q])]
                comm.send( data , dest=0)
                #print 'logelse id = ', rank, 'mean log = ', np.mean(data)

            if rank == 0:
                avg_d1 = np.mean(log)
                list_d1[0][BG.rec_count, q] = avg_d1
            list_d1[0][BG.rec_count,q] = comm.bcast(list_d1[0][BG.rec_count,q], root=0)
            comm.barrier()
            if rank == 0:
                #conn_d2 = (nest.GetConnections( source = BG.states[BG.who], target=BG.strD2[q], synapse_model='bcpnn_dopamine_synapse_d2' ))
                log = [np.log(a['p_ij']/(a['p_i']*a['p_j'])) for a in nest.GetStatus(BG.conn_d2[q]) ]
                #print 'lolog2', log, 'len = ', len(log), ' avg = ', np.mean(log)
                for i_proc in xrange(1,size):
                    log = np.r_[log, comm.recv(source=i_proc)]
            else:
                comm.send( [np.log(a['p_ij']/ (a['p_j']* a['p_i'])) for a in nest.GetStatus(BG.conn_d2[q])], dest=0)

            if rank == 0:
                avg_d2 = np.mean(log)
                list_d2[0][BG.rec_count, q] = avg_d2
            list_d2[0][BG.rec_count,q] = comm.bcast(list_d2[0][BG.rec_count,q], root=0)
            comm.barrier()

            if rank == 0:
                bias_d1 = [np.log(a['p_j']) for a in nest.GetStatus(BG.conn_d1[q]) ]
                #print 'bias_d1', bias_d1, 'len = ', len(bias_d1), ' avg = ', np.mean(bias_d1)
                for i_proc in xrange(1,size):
                    bias_d1 = np.r_[bias_d1, comm.recv(source=i_proc)]
            else:
                bias_d1 = [np.log(a['p_j']) for a in nest.GetStatus(BG.conn_d1[q]) ]
                #print 'ELSE_bias_d1', bias_d1, 'len = ', len(bias_d1), ' avg = ', np.mean(bias_d1)
                comm.send( bias_d1 , dest=0)
                #print 'logelse id = ', rank, 'mean log = ', np.mean(bias_d1)

            if rank == 0:
                avg_bias_d1 = np.mean(bias_d1)
                list_d1[8][BG.rec_count, q] = avg_bias_d1
            list_d1[8][BG.rec_count,q] = comm.bcast(list_d1[8][BG.rec_count,q], root=0)
            comm.barrier()

            if rank == 0:
                bias_d2 = [np.log(a['p_j']) for a in nest.GetStatus(BG.conn_d2[q]) ]
                #print 'bias_d2', bias_d2, 'len = ', len(bias_d2), ' avg = ', np.mean(bias_d2)
                for i_proc in xrange(1,size):
                    bias_d2 = np.r_[bias_d2, comm.recv(source=i_proc)]
            else:
                bias_d2 = [np.log(a['p_j']) for a in nest.GetStatus(BG.conn_d2[q]) ]
                #print 'ELSE_bias_d2','else id = ', rank, bias_d2, 'len = ', len(bias_d2), ' avg = ', np.mean(bias_d2)
                comm.send( bias_d2 , dest=0)

            if rank == 0:
                avg_bias_d2 = np.mean(bias_d2)
                list_d2[8][BG.rec_count, q] = avg_bias_d2
            list_d2[8][BG.rec_count,q] = comm.bcast(list_d2[8][BG.rec_count,q], root=0)
            comm.barrier()

            if rank == 0:
                print q , ' MUTAVG = ', avg_d1, 'SINGLE = ', list_d1[0][BG.rec_count,q]


#        for p in xrange(nstates):
#            conn = nest.GetConnections(source = BG.states[p], target = BG.strD1[BG.who], synapse_model = 'bcpnn_dopamine_synapse_d1')
#            sd1sw[p] = [(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]
#            conn = nest.GetConnections(source = BG.states[p], target = BG.strD2[BG.who], synapse_model = 'bcpnn_dopamine_synapse_d2')
#            sd2sw[p] = [(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]
        for n in xrange(nactions*nstates):
            #connrp = nest.GetConnections(source= BG.rp[n], target = BG.rew, synapse_model='bcpnn_dopa_synapse_RP')
            #list_rp[0][BG.rec_count,n] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(BG.conn_rp[n])])
            #list_rp[3][BG.rec_count,n] = np.mean([np.log(a['p_j']) for a in nest.GetStatus(BG.conn_rp[n])])
            if rank == 0:
                rp_bias = [np.log(a['p_j']) for a in nest.GetStatus(BG.conn_rp[n]) ]
#                print 'bias_rp', rp_bias, 'len = ', len(rp_bias), ' avg = ', np.mean(rp_bias)
                for i_proc in xrange(1,size):
                    rp_bias= np.r_[rp_bias, comm.recv(source=i_proc)]
            else:
                rp_bias = [np.log(a['p_j']) for a in nest.GetStatus(BG.conn_rp[n]) ]
#                print 'bias_rp id', rank, 'bias = ',  rp_bias, 'len = ', len(rp_bias), ' avg = ', np.mean(rp_bias)
                comm.send( rp_bias , dest=0)

            if rank == 0:
                avg_rp_bias = np.mean(rp_bias)
                list_rp[3][BG.rec_count, n] = avg_rp_bias
            list_rp[3][BG.rec_count,n] = comm.bcast(list_rp[3][BG.rec_count,n], root=0)
            comm.barrier()



            if rank == 0:
                rp_w = [np.log(a['p_ij']/(a['p_i']*a['p_j'])) for a in nest.GetStatus(BG.conn_rp[n]) ]
                #print 'w_rp', rp_w, 'len = ', len(rp_w), ' avg = ', np.mean(rp_w)
                for i_proc in xrange(1,size):
                    rp_w= np.r_[rp_w, comm.recv(source=i_proc)]
            else:
                rp_w = [np.log(a['p_ij']/(a['p_i']*a['p_j'])) for a in nest.GetStatus(BG.conn_rp[n]) ]
                #print 'w_rp id', rank, 'bias = ',  rp_w, 'len = ', len(rp_w), ' avg = ', np.mean(rp_w)
                comm.send( rp_w , dest=0)

            if rank == 0:
                avg_rp_w = np.mean(rp_w)
                list_rp[0][BG.rec_count, n] = avg_rp_w
            list_rp[0][BG.rec_count,n] = comm.bcast(list_rp[0][BG.rec_count,n], root=0)
            comm.barrier()
 

#        list_d1[0][BG.rec_count,:] =  sd1w
#        list_d2[0][BG.rec_count,:] =  sd2w
#        list_d1[8] = np.concatenate((list_d1[8], sd1sw), axis=1)
       
#        list_d2[0] = np.vstack((list_d2[0], sd2w))
#        list_d2[8] = np.concatenate((list_d2[8], sd2sw), axis=1)

#        list_br[4][BG.rec_count,:] = brw0 # = np.vstack((list_br[4], brw0))
#        list_br[5][BG.rec_count,:] = brw1 # = np.vstack((list_br[5], brw1))
#        list_br[6][BG.rec_count,:] = brw2 # = np.vstack((list_br[6], brw2))

     #   list_rp[0] = np.vstack((list_rp[0], rp_w))
        BG.rec_count += 1

    dopakf = np.r_[dopakf, dkf] 
    dopak = np.r_[dopak, dk] 
    dopam = np.r_[dopam, dm]
    dopan = np.r_[dopan, dn]
    

    return dopakf, dopak, dopam, dopan , list_d1, list_d2, list_br, list_rp
