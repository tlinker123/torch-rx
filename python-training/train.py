import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import train_funcs_ffw as tfw
import time
############################################################
####################################################################################
# Wrapper for optimizer you wish to use 
def optimizer_function(mymodel) :
	return(torch.optim.Adam(mymodel.parameters(), lr=0.01, betas=(0.9, 0.999), 

			eps=1e-08, weight_decay=0, amsgrad=False))
	#return(torch.optim.ASGD(mymodel.parameters(), lr=0.1, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0) )
####################################################################################
# Define your atomic NN as atomic_model class
# Needs to take feature_size key word as argument as input dimension size
# Currently only feature vector supported (no higher dimensional features)
####################################################################################
class atomic_model(nn.Module) :
	def __init__(self,feature_size):
		super(atomic_model, self).__init__()
		#self.flatten = nn.Flatten()
		self.linear_sig_stack = nn.Sequential(
		    nn.Linear(feature_size, 16),
		    nn.Sigmoid(),
		    #nn.Linear(20, 20),
		    #nn.Sigmoid(),
		    nn.Linear(16, 8),
		    nn.Sigmoid(),
		    nn.Linear(8, 1),
		    nn.Flatten(0,1)
		)
	def forward(self,x) :
		out=self.linear_sig_stack(x)
		return out

####################################################################################
#####################################################################################
def main() :
#####################################################################################
	#Input trainig data and other training parameters
#####################################################################################
	d=torch.load('TRAINING-DICT.pt')
	pF=100.0 #  Force weight in loss
	pE=1.0  #  Energy weight in loss
	nepochs=1000 # Number of epochs
	per_train=0.95 # percent train 
	lstart=False # Restart ?
	lVerbose=False # Output for train mse every batch iteration 
	ldump=False # Dump Predicted Energies and Forces 
	lbatch_eval_test=False # Output test mse every batch iteration, coerces lVerbose->True
	batch_size=100 # frames in batch
	MAX_NATOMS_PER_FRAME=500 # max number of atoms for any given MD frame 

#####################################################################################
# Unpacked data and global variables 
#####################################################################################
	natoms_in_frame=d['natoms_in_frame'].clone().detach()
	features=d['features'].clone().detach()
	eta=d['eta'].clone().detach()
	RS=d['RS'].clone().detach()
	RC=d['RC']
	forces=d['forces'].clone().detach()
	feature_types=d['feature_types'].clone().detach()
	features=d['features'].clone().detach()
	#features=Variable(features,requires_grad=True)
	energies=d['energies'].clone().detach()
	feature_jacobian=d['feature_jacobian']
	#feature_jacobian=torch.transpose(d['feature_jacobian'].clone().detach(),1,2)
	type_numeric=d['type_numeric'].copy()

	feature_size=eta.size()[0]*RS.size()[0]

	ntypes=len(type_numeric)
	#print(ntypes)
	types=list(type_numeric.keys())
	#print(types)
	del d

	nframes=natoms_in_frame.shape[0]
	indices_frame=torch.zeros(nframes,MAX_NATOMS_PER_FRAME,dtype=torch.int64)
	natoms_in_frame_type=torch.zeros(nframes,ntypes,dtype=torch.int64)
#####################################################################################
# frames_indices stores key mapping of feature index to MD frame
# frames_indices= [[ natoms_frames_1,index_1,index_2, ....,index_natoms_frame_1, 0 ..., MAX ],
#		    [ [ natoms_frames_2,index_1,index_2, ....,index_natoms_frame_2, 0 ..., MAX ],
#                     ...
#                     ...
# intalized in tfw.train_ffw_energy()
#####################################################################################
# Main Training Loop function
	tfw.train_ffw_energy(atomic_model,nepochs,batch_size,per_train,
	lstart,optimizer_function,indices_frame,natoms_in_frame_type,
        natoms_in_frame,pE,pF,ntypes,types,energies,forces,feature_size,features,feature_jacobian,feature_types,nframes,lVerbose,
        lbatch_eval_test, MAX_NATOMS_PER_FRAME,ldump)
#####################################################################################
if __name__ == "__main__":
    main()




