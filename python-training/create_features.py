import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from xsftools import read_xsf
import time

def main() :
####################################################################################
#Adjustable inputs 
####################################################################################
	lxsf=True   # Read from xsf file
		    # support for directly reading from QXMD and VASP output
		    # will be added soon	
		    # Extended xyz support will also be added		
	infile_xsf='./fnames'  # Path to file with xsf file names for training data
####################################################################################
	# File format is 
	# nfiles 
	# path-to-file-one 		
	# path-to-file-two
	# ...
	# ...
	# path-to-file-nfiles 		
####################################################################################
	# Feature vectors and nbrlist params
	#currently only radial feature vector
	# for PTO
####################################################################################
	eta=(torch.tensor([0.5,1.0,3.0]))
	RS = (torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
	RC=7.5
	MAXNEIGHBORS=400
	MAX_ATOMS_PER_FRAME=320
####################################################################################
	# for NH3 molecule
	#eta=(torch.tensor([0.5,1.0]))
	#RS = (torch.tensor([0.8,1.0,1.7]))
	#RC=2.5
	#MAXNEIGHBORS=10
	#MAX_ATOMS_PER_FRAME=10
####################################################################################
	#Atomic Energies are stored in atomicE to get cohesive energy
	# Character types are converted to numeric with type_numeric
	# Ex PTO 
####################################################################################
	#Pb  -72.60873748
	#Ti  -11.22479073
	#O   -3.52014750

####################################################################################
	atomicE={'Pb'  : -72.60873748,
	          'O'  :-11.22479073,
	          'Ti' :-3.52014750
		}
	type_numeric={'Pb': 0,
		       'Ti': 1,
			'O': 2 }		
####################################################################################
	# Ex NH3 molecule 
####################################################################################
	#H   -0.02679137
	#N   -0.03591556

####################################################################################
#	atomicE={'H'  : -0.02679137,
#		  'N'  :-0.03591556
#		}
#	type_numeric={'H': 0,
#		       'N': 1,
#	}
####################################################################################

	# Now code for creating and saving feature vectors 

####################################################################################
####################################################################################
####################################################################################
	# Main loop
####################################################################################
	if(lxsf) :
		tstart=time.time()
		print('Starting Feature Creation : ')
		ntypes=len(type_numeric)
		feature_size=eta.size()[0]*RS.size()[0]*ntypes
		print('Feature Size ' +str(feature_size))
		read_file=open(infile_xsf,'r')
		line=read_file.readline()
		line=line.strip().split()
		nfiles=int(line[0])
		natoms_in_frame=torch.zeros(nfiles,dtype=torch.int32)
		energies=torch.zeros(nfiles)
		forces=torch.zeros(nfiles*MAX_ATOMS_PER_FRAME,3)
		features=torch.zeros((nfiles*MAX_ATOMS_PER_FRAME,feature_size))
		feature_jacobian=torch.zeros((nfiles*MAX_ATOMS_PER_FRAME,MAX_ATOMS_PER_FRAME,feature_size,3))
		feature_types=torch.ones(nfiles*MAX_ATOMS_PER_FRAME,dtype=torch.int32)*-1
		#print('Memory Allocated for Tensor Storage' )
		#the_mem=torch.cuda.memory_allocated()
		#print(the_mem)
		cptr=0
		print('Nfiles : ' +str(nfiles))
		for i in range(nfiles) :
			line=read_file.readline()
			line=line.strip().split('\n')
			fname=line[0]
			frame=frame_data_and_features_from_xsf(fname,eta,RS,RC,MAXNEIGHBORS,atomicE,type_numeric)
			print('Packing..')
			max_feature=torch.max(frame['features'])
			min_feature=torch.min(frame['features'])
			print('Max feature : ' +str(max_feature.item()))
			print('Min feature : ' +str(min_feature.item()))
			tpack_s=time.time()
			energies[i]=frame['Energy']
			natoms_in_frame[i]=frame['natoms']
			mynatom=natoms_in_frame[i]
			for j in range(mynatom) :
				forces[cptr,:]=frame['forces'][j,:].clone().detach()
				features[cptr,:]=frame['features'][j,:].clone().detach()
				feature_jacobian[cptr,0:mynatom,:,:]=frame['feature_jacobian'][j,0:mynatom,:,:].clone().detach()
				feature_types[cptr]=frame['feature_types'][j].clone().detach()
				cptr=cptr+1
			tpack_e=time.time()
			del frame
			print('Time Pack : ' +str(tpack_e-tpack_s) +'(s)')
			print('Current Memory Allocation' )
			the_mem=torch.cuda.memory_allocated()
			print(the_mem)
		
		print('Packing and Writing to disc')
		d = {'eta' :eta, 'RS': RS,'RC':RC,  'forces': forces[0:cptr,:], 'energies': energies,
			'natoms_in_frame': natoms_in_frame, 'feature_jacobian' : feature_jacobian[0:cptr,:,:],
			'features': features[0:cptr,:],'feature_types':feature_types[0:cptr], 'type_numeric':type_numeric,	
			 'atomicE': atomicE			}
		torch.save(d,'TRAINING-DICT.pt')
		print('Done')
		tend=time.time()
		print('Total Time : ' +str(tend-tstart) +'(s)')
def frame_data_and_features_from_xsf (fname,eta,RS,RC,MAXNEIGHBORS,atomicE,type_numeric) :
	tsub=time.time()
	frame=read_xsf(fname)
	natoms=frame.natom
	types_set=list(set(frame.types))
	#print(types_set)
	ntypes=len(types_set)
	feature_types=torch.zeros(natoms,dtype=torch.int32)
	
	Eatom=0.0
	if(len(atomicE)!=ntypes) :
		raise('Atomic Energies must supplied for each type')
	for i in range(natoms) :
		mytype=frame.types[i]
		Eatom=Eatom+atomicE[mytype]
		feature_types[i]=type_numeric[mytype]
		#print(Eatom)
	myEnergy=(frame.Energy-Eatom)/natoms
	#print(feature_types)
	#myfeature_types=feature_types
	print('Cohesive Energy/natoms ' +str(myEnergy) +' eV')
	#print('ghosting .. ;)')
	#RC=7.5
	#MAXNEIGHBORS=400
	print('nbrlist ..')
	#frame.linkedlist()
	frame.get_nbr_list_ON2_pbc(MAXNEIGHBORS=MAXNEIGHBORS,RC=7.5)
	#frame.get_nbr_list_ON2(MAXNEIGHBORS=MAXNEIGHBORS)
	pos=torch.from_numpy(frame.pos)
	recip=torch.from_numpy(frame.recip)
	Hmat=torch.from_numpy(frame.Hmat)
	nbrlist=torch.from_numpy(frame.nbrlist)
	nbrlist=nbrlist.type(torch.int64)
	forces=torch.from_numpy(frame.forces)
	del frame
	#print(forces)

	ntypes=len(type_numeric)
	feature_per_type=eta.size()[0]*RS.size()[0]
	feature_size=feature_per_type*ntypes
	features=torch.zeros((natoms,feature_size),dtype=torch.float64)
	feature_jacobian=torch.zeros((natoms,natoms,feature_size,3),dtype=torch.float64)

	print('feature ..')
	get_feature_vectors_and_jacobians(recip,Hmat,natoms,eta,RC,RS,features,feature_jacobian,nbrlist,
						feature_types,ntypes,feature_per_type)
	#myfeatures=features
	#myforces=forces
	#myfeature_jacobian=feature_jacobian
	#mynatoms=natoms
	#print(features[0,:])
	#print(feature_jacobian[0,:,:])
	tend=time.time()

	print('Time nbrlist and feature comp : ' +str(tend-tsub) +'(s)')
	d = {  'forces': forces ,'Energy': myEnergy, 'feature_jacobian' : feature_jacobian,
		'features': features,'feature_types':feature_types, 'type_numeric':type_numeric	,
	 	'natoms': natoms		}
	return(d)
def get_feature_vectors_and_jacobians(recip,Hmat,natoms,eta,RC,RS,features,feature_jacobian,nbrlist,
					feature_types,ntypes,feature_per_type) :
	for i in range(natoms):
		ifeat=0
		lastneighbor=nbrlist[i,0]+1
		firstneighbor=1
		nbrlist_=nbrlist[i,firstneighbor:lastneighbor].clone().detach()
		#print(nbrlist_)
		recipj=recip[nbrlist_.tolist(),:].clone().detach()
		dr=recip[i,:]-recipj[:,:]
		#PBC still loop
		for j in range(nbrlist[i,0]) :
			for k in range(3) :
				if(dr[j,k]>0.5) :
					dr[j,k]=dr[j,k]-1.0
				elif(dr[j,k]< -0.5) :
					dr[j,k]=dr[j,k]+1.0
			dr[j,:]=torch.matmul(Hmat,dr[j,:])
			
		rij=Variable(torch.sqrt(torch.sum((dr)**2,dim=1)),requires_grad=True)
		#print(i)
		for eta_ in eta :
			for RS_ in RS :
				Radial_Feature_and_deriv(rij,dr,i,eta_,RC,RS_,nbrlist,natoms,ifeat,
					features,feature_jacobian,feature_types,ntypes,feature_per_type)
				#features[i,ifeat]=f
				#feature_jacobian[:,:,ifeat,:]=feature_jacobian[:,:,ifeat,:]+fd
				ifeat=ifeat+1
def Radial_Feature_and_deriv(rij,dr,i,eta,RC,RS,nbrlist,natoms,ifeat,features,
		feature_jacobian,feature_types,ntypes,feature_per_type) : 
	my_pi=3.14159265358979323846
	itype=feature_types[i]
	lastneighbor=nbrlist[i,0]+1
	firstneighbor=1
	nbrlist_=nbrlist[i,firstneighbor:lastneighbor].clone().detach()
	nbr_type=feature_types[nbrlist_]
	for jtype in range(ntypes) :
		stride_j=jtype*feature_per_type
	#	stride_i=itype*feature_per_type
		bool_tensor=torch.eq(nbr_type,jtype)
		rij_type=rij[bool_tensor].clone().detach()
		dr_type=dr[bool_tensor].clone().detach()
		nbrlist_type=nbrlist_[bool_tensor].clone().detach()
		rij_type.requires_grad_(requires_grad=True) 
##VECTOR IMPLEMNTATION 
		exp_rij=torch.exp(-eta*(rij_type-RS)**2)
		func_cut=0.5*(torch.cos(my_pi*rij_type/RC)+1.0)
		feature=torch.sum(exp_rij*func_cut)
		feature.backward()
		#print(ifeat+stride)
		features[i,ifeat+stride_j]=feature.clone().detach()
		rnormal=torch.div(rij_type.grad,rij_type)
		f_d=torch.transpose(torch.transpose(dr_type,0,1)*rnormal,0,1)
		f_i=torch.sum(f_d,dim=0)
		feature_jacobian[i,i,ifeat+stride_j,:]=f_i
		#print(nbrlist_type)
		feature_jacobian[i,nbrlist_type,ifeat+stride_j,:]=-1.0*f_d[:,:].clone().detach()
		#j1=0
		#for j in nbrlist_.tolist() :
		#	feature_jacobian[i,j,:]=-1.0*f_d[j1,:].clone().detach()
		#	j1=j1+1
		#rij.grad.zero_()
	#print(features[i,:])
	#print(feature_jacobian[i,0:64,ifeat,:])
	#print(feature_jacobian[i,65:320,ifeat,:])
	#raise('stop')
##LOOP IMPLEMNTATION 
#	else :
#		for j in nbrlist_.tolist() :
#			#dr=pos[i,:]-pos[j,:]
#			dr=recip[i,:]-recip[j,:]
#			for k in range(3) :
#				if(dr[k]>0.5) :
#					dr[k]=dr[k]-1.0
#				elif(dr[k]< -0.5) :
#					dr[k]=dr[k]+1.0
#			dr=torch.matmul(Hmat,dr)
#			rij=Variable(torch.sqrt(torch.sum(dr*dr)),requires_grad=True)
			#print(rij)
			#raise('rij')
#			feat=torch.exp(-eta*(rij-RS)**2)*0.5*torch.cos(rij/RC)
#			feat.backward()
#			feat_d=rij.grad*dr/rij
#			feature=feature+feat
#			feature_d[i,i,:]=feature_d[i,i,:]+feat_d
#			feature_d[i,j,:]=feature_d[i,j,:]-feat_d
	#print(feature)
	#print(feature_d)
	#rij=torch.sqrt(torch.sum((r-posj[:,:])**2,dim=1))
	#dr=r-posj[:,:]
	#print(rij)
	#feature=torch.sum(torch.exp(-eta*(rij-RS)**2)*0.5*torch.cos(rij/RC))
	#feature_d=eta*torch.exp(-eta*(rij-RS)**2)*0.5*torch.cos(rij/RC)
	return(feature)
if __name__ == "__main__":
    main()
