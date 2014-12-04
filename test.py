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

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import pylab as pl

# Load BCPNN synapse and bias-iaf neuron module                                                                                                
if (not 'bcpnn_dopamine_synapse' in nest.Models()):
    nest.sr('(/cfs/klemming/nobackup/b/berthet/code/modules/bcpnndopa_module/share/ml_module/sli) addpath') #t/tully/sequences/share/nest/sli
    nest.Install('/cfs/klemming/nobackup/b/berthet/code/modules/bcpnndopa_module/lib/nest/ml_module') #t/tully/sequences/lib/nest/pt_module

nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True})


GP = simulation_parameters.global_parameters()
GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
params = GP.params

t0 = time.time()


# ########################
# SIMULATION
# #######################


d1 = nest.Create


