import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import train_funcs_ffw_sam as tfw
import time
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--optim', type=str, default="Adam")
args = parser.parse_args()

####################################################################################
# Define your atomic NN as atomic_model class
# Needs to take feature_size key word as argument as input dimension size
# Currently only feature vector supported (no higher dimensional features)
####################################################################################


class atomic_model(nn.Module):
    def __init__(self, feature_size, Emax, Emin, FeatMax, FeatMin):
        super(atomic_model, self).__init__()
        #self.flatten = nn.Flatten()
        self.Emax = Emax
        self.FeatMax = FeatMax
        self.Emin = Emin
        self.FeatMin = FeatMin
        self.linear_sig_stack = nn.Sequential(
            nn.Linear(feature_size, 20),
            nn.Sigmoid(),
            # nn.Linear(20, 20),
            # nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 1),
            nn.Flatten(0, 1)
        )

    def bias(self, x, Max, Min):
        slope = (Max-Min)/2
        bias = Max-slope
        return(x*slope+bias)

    def biasinv(self, x, Max, Min):
        slope = 2/(Max-Min)
        bias = 1-Max*slope
        return(x*slope+bias)

    def forward(self, x):
        x = self.biasinv(x, self.FeatMax, self.FeatMin)
        x = self.linear_sig_stack(x)
        x = self.bias(x, self.Emax, self.Emin)
        return x


def main(config):
    #####################################################################################
    # Input trainig data and other training parameters
    #####################################################################################
    print('--------------------Loading DATA and Parameters------------------------------')
    d = torch.load('TRAINING-DICT-PTO.pt')
    # d = torch.load('TRAINING-DICT-NH3.pt')
    pF = 0.1  # Force weight in loss
    pE = 1.0  # Energy weight in loss
    learning_rate = 0.0005  # Overides Learning Rate in optimizer
    nepochs = 200  # Number of epochs
    per_train = 0.95  # percent train
    lstart = False  # Restart ?
    lVerbose = False  # Output for train mse every batch iteration
    ldump = True  # Dump Predicted Energies and Forces
    # Output test mse every batch iteration, coerces lVerbose->True
    lbatch_eval_test = False
    batch_size = 50  # frames in batch
    MAX_NATOMS_PER_FRAME = 325  # max number of atoms for any given MD frame

    #####################################################################################
    # Unpacked data and global variables
    #####################################################################################
    natoms_in_frame = d['natoms_in_frame'].clone().detach()
    features = d['features'].clone().detach()
    eta = d['eta'].clone().detach()
    RS = d['RS'].clone().detach()
    RC = d['RC']
    forces = d['forces'].clone().detach()
    feature_types = d['feature_types'].clone().detach()
    features = d['features'].clone().detach()
    # features=Variable(features,requires_grad=True)
    energies = d['energies'].clone().detach()
    feature_jacobian = d['feature_jacobian']
    # feature_jacobian=torch.transpose(d['feature_jacobian'].clone().detach(),1,2)
    type_numeric = d['type_numeric'].copy()
    ntypes = len(type_numeric)

    feature_size = eta.size()[0]*RS.size()[0]*ntypes

    # print(ntypes)
    types = list(type_numeric.keys())
    # print(types)
    del d

    nframes = natoms_in_frame.shape[0]
    indices_frame = torch.zeros(
        nframes, MAX_NATOMS_PER_FRAME, dtype=torch.int64)
    natoms_in_frame_type = torch.zeros(nframes, ntypes, dtype=torch.int64)
    #####################################################################################
    # frames_indices stores key mapping of feature index to MD frame
    # frames_indices= [[ natoms_frames_1,index_1,index_2, ....,index_natoms_frame_1, 0 ..., MAX ],
    #		    [ [ natoms_frames_2,index_1,index_2, ....,index_natoms_frame_2, 0 ..., MAX ],
    #                     ...
    #                     ...
    # intalized in tfw.train_ffw_energy()
    #####################################################################################
    # Main Training Loop function
    tfw.train_ffw_energy(atomic_model, learning_rate, nepochs, batch_size, per_train,
                         lstart, config.optim, indices_frame, natoms_in_frame_type,
                         natoms_in_frame, pE, pF, ntypes, types, energies, forces, feature_size, features, feature_jacobian, feature_types, nframes, lVerbose,
                         lbatch_eval_test, MAX_NATOMS_PER_FRAME, ldump)


#####################################################################################
if __name__ == "__main__":
    main(args)
