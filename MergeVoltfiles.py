import os
import numpy
import sys
import json
#import simulation_parameters

class MergeSpikefiles(object):

    def __init__(self, params):
        self.params = params


    def merge_nspike_files(self, merge_pattern, sorted_pattern_output,  sort_idx=1):
        rnd_nr1 = numpy.random.randint(0,10**8)
        rnd_nr2 = rnd_nr1 + 1
        fn_out = sorted_pattern_output
        print 'output_file:', fn_out
        # merge files from different processors
        tmp_file = "tmp_%d" % (rnd_nr2)
        os.system("cat %s_* > %s" % (merge_pattern,  tmp_file))
        # sort according to cell id
        os.system("sort -gk %d %s > %s" % (sort_idx, tmp_file, fn_out))
        os.system("rm %s" % (tmp_file))


    def merge_spiketimes_files(self, merge_pattern, sorted_pattern_output,  sort_idx=1):
        rnd_nr1 = numpy.random.randint(0,10**8)
        rnd_nr2 = numpy.random.randint(0,10**8) + 1
        fn_out = sorted_pattern_output
        print 'output_file:', fn_out
        # merge files from different processors
        tmp_file = "tmp_%d" % (rnd_nr2)
        print 'debug mergepattern', merge_pattern
        os.system("cat %s* > %s" % (merge_pattern,  tmp_file))
        # sort according to cell id
        os.system("sort -gk %d %s > %s" % (sort_idx, tmp_file, fn_out))
        os.system("rm %s" % (tmp_file))




if __name__ == '__main__':
    info_txt = \
    """
    Usage:
        python MergeSpikeFiles.py [FOLDER] [CELLTYPE] 
        or
        python MergeSpikeFiles.py [FOLDER] [CELLTYPE] [PATTERN_NUMBER]

    """
#    assert (len(sys.argv) > 2), 'ERROR: folder and cell_type not given\n' + info_txt
#   try:
#       folder = sys.argv[1]
#       params_fn = os.path.abspath(folder) + '/Parameters/simulation_parameters.json'
#       param_tool = simulation_parameters.global_parameters(params_fn=params_fn)
#   except:
#       param_tool = simulation_parameters.global_parameters()


    fparam = 'Test/Parameters/simulation_parameters.json'
    f = open(fparam, 'r')
    params = json.load(f)

#    params = param_tool.params

    cell_types_volt = ['d1', 'd2', 'actions']

    print 'nstates ', params['n_states'], 'nactions ', params['n_actions']
    MS = MergeSpikefiles(params)
    for cell_type in cell_types_volt:
        print 'Merging voltmeter data file for %s ' % (cell_type)
        print 'Merging voltmeter recordings file for %s ' % (cell_type)
        for naction in range(params['n_actions']):
            merge_pattern = params['spiketimes_folder'] + params['%s_volt_fn' % cell_type] + str(naction)
            output_fn = params['spiketimes_folder'] + str(naction) + params['%s_volt_fn_merged' % cell_type]
            MS.merge_spiketimes_files(merge_pattern, output_fn)

# need to add merging for rp and rew volt data

    cell_type = 'rew'
    print 'Merging voltmeter recordings file for %s ' % (cell_type)
    merge_pattern = params['spiketimes_folder'] + params['%s_volt_fn' % cell_type] 
    output_fn = params['spiketimes_folder'] + params['%s_volt_fn_merged' % cell_type]
    MS.merge_spiketimes_files(merge_pattern, output_fn)
        

