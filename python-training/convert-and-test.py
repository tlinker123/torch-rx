import numpy as np
import torch
from torch.autograd import Variable, grad
from torch import nn
from xsftools import read_xsf
from train import atomic_model
import time

def main() :
####################################################################################
# Inputs
# Should match how you created features
# TODO make more robust and read feature params from file, but file read is slow

####################################################################################
# If lwrite creates Torch Script Model from saved model 
# Only needs to be done once
	lwrite=False
	model_number=199 # which model 
	model_directory='./data06/' # Which directory 
####################################################################################
	infile_xsf='./fnames-testing'  # Path to file with xsf file names for training data
	eta=(torch.tensor([0.5,1.0,3.0]))
	RS = (torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
	Emin=  -2.699522018432617
	Emax= -2.4916188716888428
	FeatMax=  10.711639404296875
	FeatMin= 3.5123241663194416e-13
	RC=7.5
	MAXNEIGHBORS=400
	MAX_ATOMS_PER_FRAME=320+1
	MAX_NATOMS_PER_FRAME=320+1
	atomicE={'Pb'  : -72.60873748,
	          'O'  :-11.22479073,
	          'Ti' :-3.52014750
		}
	type_numeric={'Pb': 0,
		       'Ti': 1,
			'O': 2 }		


####################################################################################

	# Now code for indvidual frame testing

####################################################################################
####################################################################################
####################################################################################
	tstart=time.time()
	read_file=open(infile_xsf,'r')
	line=read_file.readline()
	line=line.strip().split()
	nfiles=int(line[0])
	ntypes=len(type_numeric)
	types=list(type_numeric.keys())
	feature_per_type=eta.size()[0]*RS.size()[0]
	feature_size=feature_per_type*ntypes
	models=nn.ModuleList()
	if(lwrite) :
		for itype in range(ntypes) :
			PATH=(model_directory+types[itype]+'.model.'+str(model_number).zfill(10))
			my_model=atomic_model(feature_size,torch.tensor(Emax),
							torch.tensor(Emin),
							torch.tensor(FeatMax),
							torch.tensor(FeatMin))
			checkpoint = torch.load(PATH)
			my_model.load_state_dict(checkpoint['model_state_dict'])
			compiled_model = torch.jit.script(my_model)
			PATH='./'+types[itype]+'.model_scripted.pt'
			torch.jit.save(compiled_model, PATH)
			#models.append(my_model)
	for i in range(ntypes) :
		PATH='./'+types[i]+'.model_scripted.pt'
		print(PATH)
		my_model=torch.jit.load(PATH)
		my_model.train()
		models.append(my_model)
	for i in range(nfiles) :
		tframe_s=time.time()
		line=read_file.readline()
		line=line.strip().split('\n')
		fname=line[0]
		frame=frame_data_and_features_from_xsf(fname,eta,RS,RC,MAXNEIGHBORS,atomicE,type_numeric)
		max_feature=torch.max(frame['features'])
		min_feature=torch.min(frame['features'])
		print('Max feature : ' +str(max_feature.item()))
		print('Min feature : ' +str(min_feature.item()))
		energy=torch.tensor(frame['Energy'])
		natoms_in_frame=torch.zeros((1),dtype=torch.int64)
		natoms_in_frame[0]=frame['natoms']
		natom=frame['natoms']
		#print('natom' natoms_in_frame)
		features=frame
		#mynatom=frame['natoms']
		nframes=1
		features=frame['features']
		feature_types=frame['feature_types']
		feature_jacobian=frame['feature_jacobian']
		forces=frame['forces']
		pos=frame['pos']	
		Hmat=frame['Hmat']
		H00=Hmat[0,0].item()
		H01=Hmat[0,1].item()
		H02=Hmat[0,2].item()
		H10=Hmat[1,0].item()
		H20=Hmat[2,0].item()
		H11=Hmat[1,1].item()
		H12=Hmat[1,2].item()
		H21=Hmat[2,1].item()
		H22=Hmat[2,2].item()
		#Emax=Emin=FeatMax=FeatMin=0 # Not used anymore TODO update model_energy_force batch
		tpredict_s=time.time()
		frames=torch.zeros(1,dtype=torch.int64)
		indices_frame=torch.zeros(nframes,MAX_NATOMS_PER_FRAME,dtype=torch.int64)
		natoms_in_frame_type=torch.zeros(nframes,ntypes,dtype=torch.int64)
		populate_indices_frame_and_natoms_in_frame_type(indices_frame,natoms_in_frame_type,
                        natoms_in_frame,feature_types,nframes)
		atomic_indices,energies_predict,forces_predict= model_energy_force_batch(
				   features,feature_types,
				   feature_jacobian,ntypes,frames,indices_frame,
				   natoms_in_frame,natoms_in_frame_type,models,Emax,Emin,FeatMax,FeatMin,
				   MAX_NATOMS_PER_FRAME)
		tpredict_e=time.time()
		print('Time Force Predict : ' +str(tpredict_e-tpredict_s) +'(s)')
		#print(forces_predict)
		#print(energies_predict)
		#raise('stop')
		outfile='frame.test.'+str(i).zfill(8)+'.xsf'
		write_file=open(outfile,'w')
		print('Write File opened :' + outfile)
		write_file.write("#Potential energy DFT, NN (ev/atom): ")
		write_file.write("%12.8f %12.8f\n" %(frame['Energy'],energies_predict.item()))
		write_file.write("%s\n" %("CRSYTAL"))
		write_file.write("%s\n" %("PRIMVEC"))
		write_file.write("%12.8f %12.8f %12.8f\n" %(H00,H01,H02)) 
		write_file.write("%12.8f %12.8f %12.8f\n" %(H10,H11,H12)) 
		write_file.write("%12.8f %12.8f %12.8f\n" %(H20,H21,H22))
		write_file.write("%s\n" %("PRIMCOORD"))
		write_file.write("%s %s\n" %(str(natom),str(1)))
		for iatom in range(natom):
			itype=feature_types[i].item()
			name=types[itype]
			x=pos[iatom,0].item()
			y=pos[iatom,1].item()
			z=pos[iatom,2].item()
			fx=forces[iatom,0].item()
			fy=forces[iatom,1].item()
			fz=forces[iatom,2].item()
			fx_=forces_predict[iatom,0].item()
			fy_=forces_predict[iatom,1].item()
			fz_=forces_predict[iatom,2].item()
			write_file.write("%2s  %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f" %(name,x,y,z,fx,fy,fz))
			write_file.write("%12.8f %12.8f %12.8f\n" %(fx_,fy_,fz_))
			#end for iatom
		write_file.close()
		tframe_e=time.time()
		print('Time Frame : ' +str(tframe_e-tframe_s) +'(s)')
		del frame
		del forces
		del forces_predict
		del features
		del feature_jacobian
		del indices_frame
		del Hmat
		del pos
		del natoms_in_frame
		del energy


####################################################################################
####################################################################################
#################################FUNCTIONS##########################################
####################################################################################
####################################################################################
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
	 	'natoms': natoms , 'Hmat': Hmat, 'pos': pos		}
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
def populate_indices_frame_and_natoms_in_frame_type(indices_frame,natoms_in_frame_type,
			natoms_in_frame,feature_types,nframes) :
#############################################################
# frames_indices stores key mapping of feature index to MD frame
# frames_indices= [[ natoms_frame_1,index_1,index_2, ....,index_natoms_frame_1, 0 ..., MAX ],
#		    [ [ natoms_frame_2,index_1,index_2, ....,index_natoms_frame_2, 0 ..., MAX ],
#                     ...
#                     ...
#
#######################################################################################
	i=0
	# Probably more efficent way to populate using torch.arange then double loop
	# But at most few 100,000 atoms in train set and only called once
	for j in range(nframes) : 
		indices_frame[j,0]=natoms_in_frame[j]
		for k in range(natoms_in_frame[j]) :
			indices_frame[j,1+k]=i
			idx=feature_types[i]
			natoms_in_frame_type[j,idx]=natoms_in_frame_type[j,idx]+1
			i=i+1
			
		
#############################################################
def get_indices_type_frames(indices_frame,mytype,frames,feature_types) :
#############################################################
	# Get atomic indices corresponding to specfic frame for atom type mytype
	# If mytype<0 then indices for entire frame are returned 
	if( len(frames.size())>0) :
		indices=indices_frame[frames[0],1:(indices_frame[frames[0],0]+1)].clone()
		for k in range(1,frames.size()[0]) :
			indices=torch.cat((indices,indices_frame[frames[k],1:(indices_frame[frames[k],0]+1)].clone()))
		if(mytype>=0) :
			return(indices[torch.eq(feature_types[indices],mytype)].clone())
		else  :
			return(indices)
	else :
		indices=indices_frame[frames.item(),1:(indices_frame[frames.item(),0]+1)].clone()
		if(mytype>=0) :
			return(indices[torch.eq(feature_types[indices],mytype)].clone())
		else  :
			return(indices)

#############################################################
def sum_tensor_parts(t, lens):
#############################################################
# Output from NN model for atom type k will be 
# [Eatom_k1_frame1, Eatom_k2_frame1, ...,E_atom_kn_frame1,E_atom_k1_frame2, ...
# We need partial sums over each frame for output 
# [E_frame_1,E_frame_2,...,E_frame_N] 
# 
#############################################################
# t : input tensor 
# lens : lenghts to sum  over 
# ex t=tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 1.])
#    lens= tensor([ 8, 12])
#    output= tensor([8., 3.]) 	
#############################################################
#see https://stackoverflow.com/questions/71800953/how-can-i-sum-parts-of-pytorch-tensor-of-variable-sizes
#############################################################
    t_size_0 = t.size(0)
    ind_x = torch.repeat_interleave(torch.arange(lens.size(0)), lens)
    indices = torch.cat(
        [
            torch.unsqueeze(ind_x, dim=0),
            torch.unsqueeze(torch.arange(t_size_0), dim=0)
        ],
        dim=0
    )
    M = torch.sparse_coo_tensor(
        indices,
        torch.ones(t_size_0, dtype=torch.float32),
        size=[lens.size(0), t_size_0]
    )
    return M @ t

def model_energy_force_batch(features,feature_types,feature_jacobian,ntypes,frames,indices_frame,
				natoms_in_frame,natoms_in_frame_type,models,Emax,Emin,FeatMax,FeatMin,
					MAX_NATOMS_PER_FRAME) :
#############################################################
	# Storage for predicted force and energy
	# Forces appended with tensor.cat() to avoid implace ops
	# predicted summed for each atom type and added vector wise
	f_out=torch.zeros((1,3),requires_grad=True)
	e_out=torch.zeros(frames.size()[0])
	feature_size=features.size()[1]
	# storing tables for gradients for each atom type
	# Each gradient must be stored for each atom to compute force
	# graidents stored with tensor.cat to avoid implace ops
	gradient=torch.zeros((1,feature_size))
	nframes=frames.size()[0]
	gradient_frames=torch.zeros(nframes,MAX_NATOMS_PER_FRAME,dtype=torch.int64)
	start_type=torch.zeros(nframes,ntypes,dtype=torch.int64)
	end_type=torch.zeros(nframes,ntypes,dtype=torch.int64)
	batch_natoms_in_frame_type=natoms_in_frame_type[frames,:]
	for l in range(nframes) :
		start=0
		for k in range(ntypes):
			start_type[l,k]=start
			end=start+batch_natoms_in_frame_type[l,k]
			end_type[l,k]=end
			start=start+batch_natoms_in_frame_type[l,k]	
	
	start=0
	end=0
	#print(start_type)
	#print(end_type)
	#raise('stop')
	for k in range(ntypes) :
		#print(k)
		#############################################################
		# Forward pass
		indices=get_indices_type_frames(indices_frame,k,frames,feature_types)
		x_in=Variable(features[indices,:].clone(),requires_grad=True)
		#x2=biasinv(x_in,FeatMax,FeatMin)
		y_pred=models[k](x_in.float())
		#y_pred=bias(y_pred,Emax,Emin)
		#############################################################
		# Storing frame lookup  for gradient
		batch_natoms_in_frame=natoms_in_frame_type[frames,k]
		for l in range(nframes) :
			i1=start_type[l,k].item()
			i2=end_type[l,k].item()
			end=end+batch_natoms_in_frame_type[l,k]
			#print(i1)
			#print(i2)
				#print(torch.arange(start,end)).size()
			gradient_frames[l,i1:i2]=torch.arange(start,end)
			start=end
		#print(gradient_frames)
		#############################################################
		# Compute gradient, create_graph=True is requried for force to contribute to loss
		mygrad=grad(y_pred, inputs=x_in, grad_outputs=torch.ones_like(y_pred) ,retain_graph=True, create_graph=True)
		#############################################################
		# Sum atomic energies for each frame and store gradients
	
		# sum_tensor_parts sums over lenghts specified in batch_natoms_in_frame
		# Thus a tensor is returned with partial sums over each atom within each MD frame
		e_out=e_out+sum_tensor_parts(y_pred, batch_natoms_in_frame)

		gradient=torch.cat((gradient,mygrad[0]))
		#############################################################
		#x_in.grad.zero_()
	#raise('stop')
	gradient=gradient[1:gradient.size()[0]]
	#print(gradient.size())
	#print(e_out)
	#print(natoms_in_frame[frames])
	e_out=e_out/natoms_in_frame[frames]
	#print(e_out)
	#raise('test')
	for k in range(frames.size()[0]) :
		indices_my_atoms=get_indices_type_frames(indices_frame,-1,frames[k],feature_types)
		natoms=natoms_in_frame[frames[k]]
		#print(gradient_frames[k,0:natoms])
		mygradient=gradient[gradient_frames[k,0:natoms],:]
		#mygradient=torch.zeros((natoms,feature_size))
		myjac=feature_jacobian[indices_my_atoms,0:natoms,:,:]
		# F_ik=-dG_jn/dr_ik *dE_j/dG_jn
		# O(N^2) Tensor Product, but faster than loops in python lol
		# One should do neighbor list loop, but this will require inplace ops
		myforces=-1.0*torch.tensordot(myjac,mygradient,dims=([0,2],[0,1]))
		f_out=torch.cat((f_out,myforces))
	f_out=f_out[1:f_out.size()[0],:]
	#print(f_out)
	#raise('stop')
	index_out=get_indices_type_frames(indices_frame,-1,frames,feature_types)
	return(index_out,e_out,f_out)
if __name__ == "__main__":
    main()
