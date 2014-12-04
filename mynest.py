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

    conn_dopa = nest.GetConnections(BG.states[0], BG.strD1[BG.who], synapse_model='bcpnn_dopamine_synapse_d1' )
    conn_dopa2 = nest.GetConnections(BG.states[0], BG.strD2[BG.who], synapse_model='bcpnn_dopamine_synapse_d2' )




    sim_time = time / resolution 
    dkf = np.empty(sim_time)
    dk = np.empty(sim_time)
    dm = np.empty(sim_time)
    dn = np.empty(sim_time)

    sd1pi = np.empty(nactions)
    sd1pj = np.empty(nactions)
    sd1pij = np.empty(nactions)
    sd1eij = np.empty(nactions)
    sd1eijc = np.empty(nactions)
    sd1w = np.empty(nactions)

    sd2pi = np.empty(nactions)
    sd2pj = np.empty(nactions)
    sd2pij = np.empty(nactions)
    sd2eij = np.empty(nactions)
    sd2eijc = np.empty(nactions)
    sd2w = np.empty(nactions)

    ad1w = np.empty(nstates)
    sd1sw = np.empty((nstates, len(conn_dopa)))
    ad2w = np.empty(nstates)
    sd2sw = np.empty((nstates, len(conn_dopa2)))
    std1w = np.empty(nstates)
    std2w = np.empty(nstates)
    
    brpi = np.empty(nactions)
    brpj = np.empty(nactions)
    brpij = np.empty(nactions)
    breij = np.empty(nactions)
    brw0 = np.empty(nactions)
    brw1 = np.empty(nactions)
    brw2 = np.empty(nactions)

    rp_w = np.empty(nactions*nstates)
    rp_pi = np.empty(nactions*nstates)
    rp_pj = np.empty(nactions*nstates)
    rp_pij = np.empty(nactions*nstates)
    
    for i in xrange(int(sim_time)):
        nest.Simulate(resolution)
        dkf[i], dk[i], dm[i], dn[i] =  nest.GetStatus(conn_dopa2, ['k_filtered', 'k', 'm', 'n'])[0]
        for q in xrange(nactions):
            conn = nest.GetConnections(source = BG.states[BG.who], target = BG.strD1[q], synapse_model = 'bcpnn_dopamine_synapse_d1')
            sd1pi[q] = np.mean( [a['p_i'] for a in nest.GetStatus(conn)] ) 
            sd1pj[q] = np.mean( [a['p_j'] for a in nest.GetStatus(conn)] ) 
            sd1pij[q] = np.mean( [a['p_ij'] for a in nest.GetStatus(conn)] )
            sd1eij[q] = np.mean( [a['e_ij'] for a in nest.GetStatus(conn)] )
            sd1eijc[q] = np.mean( [a['e_ij_c'] for a in nest.GetStatus(conn)] )
            sd1w[q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            #print'CONNSIZE D1: ', len(conn)
            #pp.pprint(conn)
            conn = nest.GetConnections(source = BG.states[BG.who], target = BG.strD2[q], synapse_model = 'bcpnn_dopamine_synapse_d2')
            sd2pi[q] = np.mean( [a['p_i'] for a in nest.GetStatus(conn)] ) 
            sd2pj[q] = np.mean( [a['p_j'] for a in nest.GetStatus(conn)] ) 
            sd2pij[q] = np.mean( [a['p_ij'] for a in nest.GetStatus(conn)] ) 
            sd2eij[q] = np.mean( [a['e_ij'] for a in nest.GetStatus(conn)] ) 
            sd2eijc[q] = np.mean( [a['e_ij_c'] for a in nest.GetStatus(conn)] ) 
            sd2w[q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            #print 'CONNSIZE D2: ', len(conn)
            #pp.pprint(conn)
            
            conn = nest.GetConnections(source = BG.states[BG.who], target = BG.brainstem[q], synapse_model = 'bcpnn_synapse')
            brpi[q] = np.mean( [a['p_i'] for a in nest.GetStatus(conn)] ) 
            brpj[q] = np.mean( [a['p_j'] for a in nest.GetStatus(conn)] ) 
            brpij[q] = np.mean( [a['p_ij'] for a in nest.GetStatus(conn)] ) 
            #breij[q] = np.mean( [a['e_ij'] for a in nest.GetStatus(conn)] ) 
            brw0[q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            conn = nest.GetConnections(source = BG.states[1], target = BG.brainstem[q], synapse_model = 'bcpnn_synapse')
            brw1[q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            conn = nest.GetConnections(source = BG.states[2], target = BG.brainstem[q], synapse_model = 'bcpnn_synapse')
            brw2[q] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
        for p in xrange(nstates):
            conn = nest.GetConnections(source = BG.states[p], target = BG.strD1[BG.who], synapse_model = 'bcpnn_dopamine_synapse_d1')
            ad1w[p] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            std1w[p] = np.std([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            sd1sw[p] = [(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]
            #sd1sw[p] = [(a['p_i']) for a in nest.GetStatus(conn)]
            
            conn = nest.GetConnections(source = BG.states[p], target = BG.strD2[BG.who], synapse_model = 'bcpnn_dopamine_synapse_d2')
            ad2w[p] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            std2w[p] = np.std([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            sd2sw[p] = [(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]
            #sd2sw[p] = [(a['p_i']) for a in nest.GetStatus(conn)]
        
        for n in xrange(nactions*nstates):
            connrp = nest.GetConnections(source= BG.rp[n], target = BG.rew, synapse_model='bcpnn_dopa_synapse_RP')
            rp_w[n] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(connrp)])
            rp_pi[n] = np.mean([a['p_i'] for a in nest.GetStatus(connrp)])
            rp_pj[n] = np.mean([a['p_j'] for a in nest.GetStatus(connrp)])
            rp_pij[n] = np.mean([a['p_ij'] for a in nest.GetStatus(connrp)])

 

        list_d1[0] = np.vstack((list_d1[0], sd1w))
        list_d1[1] = np.vstack((list_d1[1], ad1w))
        list_d1[2] = np.vstack((list_d1[2], sd1pi))
        list_d1[3] = np.vstack((list_d1[3], sd1pj))
        list_d1[4] = np.vstack((list_d1[4], sd1pij))
        list_d1[5] = np.vstack((list_d1[5], sd1eij))
        list_d1[6] = np.vstack((list_d1[6], sd1eijc))
        list_d1[7] = np.vstack((list_d1[7], std1w))
        list_d1[8] = np.concatenate((list_d1[8], sd1sw), axis=1)
       
        list_d2[0] = np.vstack((list_d2[0], sd2w))
        list_d2[1] = np.vstack((list_d2[1], ad2w))
        list_d2[2] = np.vstack((list_d2[2], sd2pi))
        list_d2[3] = np.vstack((list_d2[3], sd2pj))
        list_d2[4] = np.vstack((list_d2[4], sd2pij))
        list_d2[5] = np.vstack((list_d2[5], sd2eij))
        list_d2[6] = np.vstack((list_d2[6], sd2eijc))
        list_d2[7] = np.vstack((list_d2[7], std2w))
        list_d2[8] = np.concatenate((list_d2[8], sd2sw), axis=1)

        #list_br[0] = np.vstack((list_br[0], breij))
        list_br[1] = np.vstack((list_br[1], brpi))
        list_br[2] = np.vstack((list_br[2], brpj))
        list_br[3] = np.vstack((list_br[3], brpij))
        list_br[4] = np.vstack((list_br[4], brw0))
        list_br[5] = np.vstack((list_br[5], brw1))
        list_br[6] = np.vstack((list_br[6], brw2))
        #print 'DEBUGNAN', sd1w, ad1w, sd1pi, sd1pj, sd1pij, sd1eij, sd1eijc, sd2w, ad2w, sd2pi, sd2pj, sd2pij, sd2eij, sd2eijc

        list_rp[0] = np.vstack((list_rp[0], rp_w))
        list_rp[1] = np.vstack((list_rp[1], rp_pi))
        list_rp[2] = np.vstack((list_rp[2], rp_pj))
        list_rp[3] = np.vstack((list_rp[3], rp_pij))

    dopakf = np.r_[dopakf, dkf] 
    dopak = np.r_[dopak, dk] 
    dopam = np.r_[dopam, dm]
    dopan = np.r_[dopan, dn]

    return dopakf, dopak, dopam, dopan , list_d1, list_d2, list_br, list_rp
    

