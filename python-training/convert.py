import torch 
import numpy as np
import torch
from torch import nn
from train import atomic_model
import atomic_constants as ac
#########################################################################
# This Script converts model to torch-script to be loaded
# by C++ API for on the fly force and energy prediciton in MD simmulation
#
# This script also saves feature parameters (eta,RS,RC,...) to be read
# by MD driver for on the fly feature creation  
##########################################################################


def main() :
##########################################################################
	# Input Parameters
	d=torch.load('TRAINING-DICT.pt') 
	model_number=199 # which model 
	model_directory='./data06/' # Which directory 

	#Write Feature Parameters to input file for MD simmulation
	##########################################################################

	eta=d['eta'].clone().detach()
	RS=d['RS'].clone().detach()
	RC=d['RC']
	type_numeric=d['type_numeric'].copy()
	ntypes=len(type_numeric)

	feature_size=eta.size()[0]*RS.size()[0]*ntypes

	#print(ntypes)
	types=list(type_numeric.keys())
	param_file=open('FEATURE-PARAM.in','w')
	eta_size=eta.size()[0]
	param_file.write('FEATURE PARAMETERS\n\n')
	param_file.write('ETA OUTER LOOP, RS INNER LOOP\n\n')
	param_file.write('(MODEL_TYPE_INFO)\n')
	param_file.write('%6i\n' %(ntypes))
	for itype in range(ntypes) :
		mytype_up=types[itype].upper()
		index=ac.aname.index(mytype_up)
		mymass=ac.dmassn[index]
		fname=types[itype]+'.model_scripted.pt'
		param_file.write('%s %12.6f %s\n' %(types[itype], mymass, fname))
	
	
	param_file.write('(ETA_RADIAL)\n')
	param_file.write('%6i\n' %(eta_size))
	for i in range(eta_size)  :
		param_file.write( '%12.6f' %(eta[i]))
	param_file.write('\n')
	RS_size=RS.size()[0]
	param_file.write('(RS_RADIAL)\n')
	param_file.write('%6i\n' %(RS_size))
	for i in range(RS_size)  :
		param_file.write( '%12.6f' %(RS[i]))
	param_file.write('\n')
	bias_file=open(model_directory+'bias.txt','r')
	mylist=list()
	for i in range(4) :
		line=bias_file.readline()
		line=line.strip().split()
		s1='('+line[0]+'_in_train)'
		s2=line[1]
		mylist.append(float(s2))
		param_file.write( '%s\n' %(s1))
		param_file.write( '%s\n' %(s2))
	
	Emin=mylist[0]
	Emax=mylist[1]
	FeatMax=mylist[2]
	FeatMin=mylist[3]
	PATH=(model_directory+types[itype]+'.model.'+str(model_number).zfill(10))
	for itype in range(ntypes) :
		my_model=atomic_model(feature_size,torch.tensor(Emax),
						torch.tensor(Emin),
						torch.tensor(FeatMax),
						torch.tensor(FeatMin))
		checkpoint = torch.load(PATH)
		my_model.load_state_dict(checkpoint['model_state_dict'])
		model_scripted = torch.jit.script(my_model) # Export to TorchScript
		model_scripted.save(types[itype]+'.model_scripted.pt') # Save
	


if __name__ == "__main__":
    main()
