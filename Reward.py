import numpy as np

class Reward(object):

    def __init__(self, params):
        self.params = params

# need to pay attention to MPI implementation if random
    def compute_reward(self, state, action, block):
        rew = 0
        if action == ((block + state) % self.params['n_actions']):
            rew = 1
        print 'SUMMARY state %d, action %d, block %d, gloubi %d' % (state, action, block, block+state)
        return rew

    def compute_reward_2(self, state, action, block):
        rew = 0
        if action == ((block%2 + state) % self.params['n_actions']):
            rew = 1
        print 'SUMMARY state %d, action %d, block %d, gloubi %d' % (state, action, block, block+state)
        return rew
