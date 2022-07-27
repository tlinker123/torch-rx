import numpy as np
import torch
from torch.autograd import Variable, grad
from torch import nn
import time
#from energymodel import atomic_model,atomic_model_normal
########################################################################################
# Main Training Loop function
########################################################################################
def train_ffw_energy(atomic_model,nepochs,batch_size,per_train,lstart,optimizer_function,indices_frame,natoms_in_frame_type,
	natoms_in_frame,pE,pF,ntypes,types,energies,forces,feature_size,features,feature_jacobian,feature_types,nframes,lVerbose,
	lbatch_eval_test,MAX_NATOMS_PER_FRAME,ldump) :
########################################################################################
# Initalize Log files and print inputs to logs and stderr.
	tmaster=time.time()
	print('------------------------------------------------------')
	print('----------------TORCH RX HAS STARTED------------------')
	print('')
	
	log_file=open('./TRAINING-LOG.txt','w')
	RSME_file=open('./TRAINING-RSME.txt','w')
	time_file=open('./TRAINING-TIMING.txt','w')
	time_file.close()
	RSME_file.write('RSME Log for NN Training\n')
	RSME_file.write('Units are in eV/atom and ev/Angstrom for energy and force RSME\n')
	RSME_file.write('------------------------------------------------------')
	RSME_file.write('------------------------------------------------------\n')
	RSME_file.write("%s %s %s %s %s %s %s %s %s\n " %('epoch', 'ERSME-TRAIN' ,'CEx-Train' ,  
				 'FRSME-TRAIN', 'CFx-Train' ,
				'ERSME-TEST', 'CEx-Test',  'FRSME-TEST', 'CFx-TEST' ))
	RSME_file.write('------------------------------------------------------')
	RSME_file.write('------------------------------------------------------\n')
	RSME_file.write('\n')
	log_file.close()
	RSME_file.close()
	#torch.autograd.set_detect_anomaly(True) # For Debugging, 10X slowdown if true
	#torch.set_num_interop_threads(4) # Inter-op parallelism
	#torch.set_num_threads(4) 
	bias_file=open('./data/bias.txt','w')
	Emin=torch.min(energies).item()
	Emax=torch.max(energies).item()
	FeatMax=torch.max(features).item()
	FeatMin=torch.min(features).item()
	bias_file.write('EMIN  ' + str(Emin)+'\n')
	bias_file.write('EMAX ' + str(Emax)+'\n' )
	bias_file.write('FeatMax  ' + str(FeatMax) +'\n')
	bias_file.write('FeatMin ' + str(FeatMin)+'\n' )
	bias_file.close()
	#feature_jacobian=torch.transpose(feature_jacobian,1,2)
	#print(features.size())
	print('Starting training for energy model with force prediction')
	print('------------------------------------------------------')
	print('Emin : ' + str(Emin) + ' eV/atom')
	print('Emax : ' + str(Emax) + ' eV/atom')
	print('FeatMax : ' + str(FeatMax) )
	print('FeatMin : ' + str(FeatMin) )
	print('pF (force contribution to loss) : ' +str(pF)) 
	print('pE (energy contribution to loss) : ' +str(pE)) 
	print('nepochs : ' +str(nepochs))
	print('per_train : ' +str(per_train))
	print('lstart : ' +str(lstart))
	print('lVerbose : ' +str(lVerbose))
	print('batch_size : ' +str(batch_size))
	print('MAX_NATOMS_PER_FRAME : ' +str(MAX_NATOMS_PER_FRAME))
	log_file=open('./TRAINING-LOG.txt','a')
	log_file.write('------------------------------------------------------\n')
	log_file.write('----------------TORCH RX HAS STARTED------------------\n\n')
	log_file.write('Starting training for energy model with force prediction\n')
	log_file.write('------------------------------------------------------\n')
	log_file.write('------------------------------------------------------\n')
	log_file.write('Emin : ' + str(Emin) + ' eV/atom\n')
	log_file.write('Emax : ' + str(Emax) + ' eV/atom\n')
	log_file.write('FeatMax : ' + str(FeatMax) +'\n' )
	log_file.write('FeatMin : ' + str(FeatMin) +'\n' )
	log_file.write('pF (force contribution to loss) : ' +str(pF)+'\n') 
	log_file.write('pE (energy contribution to loss) : ' +str(pE)+'\n') 
	log_file.write('nepochs : ' +str(nepochs)+'\n')
	log_file.write('per_train : ' +str(per_train)+'\n')
	log_file.write('lstart : ' +str(lstart)+'\n')
	log_file.write('lVerbose : ' +str(lVerbose)+'\n')
	log_file.write('batch_size : ' +str(batch_size)+'\n')
	log_file.write('MAX_NATOMS_PER_FRAME : ' +str(MAX_NATOMS_PER_FRAME)+'\n')
	log_file.close()
########################################################################################
# Initalize predicted array for predicted atomic forces and MD frame energies 
	all_forces_predict=forces.clone().detach() *0.0
	all_energies_predict=energies.clone().detach() *0.0
########################################################################################
# Populate key that maps frame number to indices of feature vectors in that frame
	populate_indices_frame_and_natoms_in_frame_type(indices_frame,natoms_in_frame_type,
                        natoms_in_frame,feature_types,nframes)
########################################################################################
# Load/Initalize models and test/train frames
# Models store in nn.module list
	if not lstart :
		shuffled_frames=torch.randperm(nframes,dtype=torch.int64)
		ntrain=int(np.floor(per_train*nframes))
		train=shuffled_frames[0:ntrain].clone().detach()
		test=shuffled_frames[(ntrain):nframes].clone().detach()
		del shuffled_frames
		torch.save({'train':train,'test':test} ,'FRAMES-SAVED.pt' )
		models=nn.ModuleList()
		#optimizers=list()
		print('ntypes : ' +str(ntypes))
		print('types : ' +str(types))
		log_file=open('./TRAINING-LOG.txt','a')
		log_file.write('ntypes  : ' + str(ntypes)+'\n')
		log_file.write('types  : ' + str(types)+'\n\n\n')
		for i in range(ntypes) :
			#mymodel=atomic_model()
			#models.append(atomic_model_normal(feature_size,Emin,Emax))
			models.append(atomic_model(feature_size))
			print('Initalized model for : '+str(types[i]))
			print(models[i])
			log_file.write('Initalized model for : '+str(types[i])+'\n\n')
			log_file.write(str(models[i]))
			log_file.write('\n\n')
			#optimizers.append(optimizer_function(models[i]))
		optimizer=optimizer_function(models)
		log_file.close()
	else :
		#return 
		loaded_frames=torch.load('FRAMES-SAVED.pt')
		train=loaded_frames['train'].clone().detach()
		test=loaded_frames['test'].clone().detach()
		del loaded_frames
		models=nn.ModuleList()
		#optimizers=list()
		print('ntypes : ' +str(ntypes))
		print('types : ' +str(types))
		log_file=open('./TRAINING-LOG.txt','a')
		log_file.write('ntypes  : ' + str(ntypes)+'\n')
		log_file.write('types  : ' + str(types)+'\n\n\n')
		for i in range(ntypes) :
			#mymodel=atomic_model()
			#models.append(atomic_model_normal(feature_size,Emin,Emax))
			my_model=torch.load('./'+types[i]+'.model.RESTART')
			models.append(my_model)
			print('Loaded  model for : '+str(types[i]))
			print(models[i])
			log_file.write('Loaded model for : '+str(types[i])+'\n\n')
			log_file.write(str(models[i]))
			log_file.write('\n\n')
			#optimizers.append(optimizer_function(models[i]))
		optimizer=optimizer_function(models)
		log_file.close()
########################################################################################
# Output train/test set decomposition 
	nframes_train=train.size()[0]
	nframes_test=test.size()[0]
	print('nframes train : '+ str(nframes_train))
	print(train)
	print('nframes test : '+ str(nframes_test))
	print(test)
	log_file=open('./TRAINING-LOG.txt','a')
	log_file.write('nframes train : '+ str(nframes_train)+'\n')
	log_file.write(str(train)+'\n')
	log_file.write('nframes test : '+ str(nframes_test)+'\n')
	log_file.write(str(test)+'\n\n')
	tend=time.time()
	print('Time Load : ' +str(tend-tmaster) +'(s)')
	log_file.write('Time Load : ' +str(tend-tmaster) +'(s)\n\n')
	log_file.write('END INFO and PARAMETERS')
	log_file.close()
########################################################################################
# Main Training Loop
	for i in range(nepochs) :
		tepoch=time.time()
		lossfn=nn.MSELoss()
		my_ESME=0.0
		my_FSME=0.0
		print('Epoch ' +str(i) )
		print('-----------------Run Batches---------------------')
		batch_j=0
		for j in range(0,nframes_train,batch_size) :
			if(lVerbose) :
				print('Epoch ' +str(i) +' Batch ' +str(batch_j))
			frames=train[j:j+batch_size].clone()
			optimizer.zero_grad(set_to_none=False)
		####################################################################################
		# Evaluate energies,forces.
			atomic_indices,energies_predict,forces_predict= model_energy_force_batch(
					   features,feature_types,
			                   feature_jacobian,ntypes,frames,indices_frame,
                                	   natoms_in_frame,natoms_in_frame_type,models,Emax,Emin,FeatMax,FeatMin,
					   MAX_NATOMS_PER_FRAME)
		####################################################################################
		# Store predicted energies,forces.
			all_energies_predict[frames]=energies_predict.clone().detach()
			all_forces_predict[atomic_indices,:]=forces_predict.clone().detach()
		####################################################################################
		# Compute Loss
			loss,ESME,FSME=loss_Force_Energy(Ein=energies_predict,Fin=torch.flatten(forces_predict),
				Etarget=energies[frames],Ftarget=torch.flatten(forces[atomic_indices,:]),
				natoms_in_frame=natoms_in_frame[frames],nframes_batch=frames.size()[0],
				pF=pF,pE=pE,
				lVerbose=(lVerbose or lbatch_eval_test),mode='TRAIN', lossfn=lossfn)
		####################################################################################
		#  Accumlate Loss for batch
			my_ESME=my_ESME+ESME*frames.size()[0]
			my_FSME=my_FSME+FSME*frames.size()[0]
			del frames
		####################################################################################
		#  Update Weights
			optimizer.zero_grad(set_to_none=False)
			loss.backward()
			optimizer.step()
			batch_j=batch_j+1
		####################################################################################
		#End Loop over Batches 
		####################################################################################
		# Output RMSE for entire train
		print('-----------------Train Eval---------------------')
		mode='TRAIN'
		frames=train.clone()
		my_ESME=my_ESME/frames.size()[0]
		my_FSME=my_FSME/frames.size()[0]
		my_ERSME=np.sqrt(my_ESME)
		my_FRSME=np.sqrt(my_FSME)
		print(all_energies_predict[20])
		print(energies[20])
		print(all_forces_predict[20,:])
		print(forces[20,:])
		print(mode+' --> ' + 'RMSE Energy : ' +str(my_ERSME)+' eV/atom')
		print(mode+' --> ' + 'RMSE Force : ' +str(my_FRSME)+' eV/A')
		atomic_indices=get_indices_type_frames(indices_frame,-1,frames,feature_types)
		CEx=pearsoncorr(all_energies_predict[frames],energies[frames])
		CFx=pearsoncorr(torch.flatten(all_forces_predict[atomic_indices,:]),
					torch.flatten(forces[atomic_indices,:]))
		if(ldump) :
			torch.save(all_energies_predict[frames],'./data/train-eng-pred.'+str(i).zfill(10)+'.pt')
			torch.save(torch.flatten(all_forces_predict[atomic_indices,:]),
					'./data/train-frc-pred.'+str(i).zfill(10)+'.pt')
			if(i<1) : 
				torch.save(energies[frames],'./data/train-eng-ground-truth.pt')
				torch.save(torch.flatten(forces[atomic_indices,:]),
						'./data/train-frc-ground-truth.pt')
		CEx_Train=CEx.item()
		CFx_Train=CFx.item()
		print(mode+' --> ' + 'CEx : ' +str(CEx.item()))
		print(mode+' --> ' + 'CFx : ' +str(CFx.item()))
		####################################################################################
		# Compute RMSE for test set
		print('-----------------Test Eval---------------------')
		mode='TEST'
		frames=test.clone()
		#atomic_indices=get_indices_type_frames(indices_frame,-1,frames,feature_types)
		atomic_indices,energies_predict,forces_predict= model_energy_force_batch(
				   features,feature_types,
				   feature_jacobian,ntypes,frames,indices_frame,
				   natoms_in_frame,natoms_in_frame_type,models,Emax,Emin,FeatMax,FeatMin,
					MAX_NATOMS_PER_FRAME)
		#optimizer.zero_grad(set_to_none=False)
		loss,test_ESME,test_FSME=loss_Force_Energy(Ein=energies_predict,Fin=torch.flatten(forces_predict),
			Etarget=energies[frames],Ftarget=torch.flatten(forces[atomic_indices,:]),
			natoms_in_frame=natoms_in_frame[frames],nframes_batch=frames.size()[0],
			pF=pF,pE=pE,
			lVerbose=False,mode='TEST', lossfn=lossfn)
		test_ERSME=np.sqrt(test_ESME)
		test_FRSME=np.sqrt(test_FSME)
		print(mode+' --> ' + 'RMSE Energy : ' +str((test_ERSME))+' eV/atom')
		print(mode+' --> ' + 'RMSE Force : ' +str((test_FRSME))+' eV/A')
		CEx=pearsoncorr(energies_predict,energies[frames])
		CFx=pearsoncorr(torch.flatten(forces_predict),
					torch.flatten(forces[atomic_indices,:]))
		if(ldump) :
			torch.save(energies_predict,'./data/test-eng-pred.'+str(i).zfill(10)+'.pt')
			torch.save(torch.flatten(forces_predict),
					'./data/test-frc-pred.'+str(i).zfill(10)+'.pt')
			if(i<1) : 
				torch.save(energies[frames],'./data/test-eng-ground-truth.pt')
				torch.save(torch.flatten(forces[atomic_indices,:]),
						'./data/test-frc-ground-truth.pt')
		CEx_Test=CEx.item()
		CFx_Test=CFx.item()
		print(mode+' --> ' + 'CEx : ' +str(CEx.item()))
		print(mode+' --> ' + 'CFx : ' +str(CFx.item()))
		####################################################################################
		# Output RMSE to file and save models
		RSME_file=open('./TRAINING-RSME.txt','a')
		RSME_file.write("%6i %.6E %.6E %.6E %.6E %.6E %.6E %.6E %.6E\n " 
						%(i, my_ERSME,CEx_Train,my_FRSME,CFx_Train,
							test_ERSME, CEx_Test, test_FRSME, CFx_Test))
		RSME_file.close()
		for itype in range(ntypes) :
			torch.save(models[itype],'./data/'+types[itype]+'.model.'+str(i).zfill(10))
		print('-----------------------------------------------')
		del frames
		del loss
		time_file=open('./TRAINING-TIMING.txt','a')
		tend=time.time()
		print('Time Epoch : ' +str(tend-tepoch) +'(s)')
		print('-----------------------------------------------')
		time_file.write("%6i %12.6f\n" %(i, tend-tepoch ))
		time_file.close()
		####################################################################################
	# End Loop over epochs 
	time_file=open('./TRAINING-TIMING.txt','a')
	tend=time.time()
	print('Total Time : ' +str(tend-tmaster) +'(s)')
	time_file.write("\n\n")
	time_file.write("%s %12.6f" %('total', tend-tmaster ))

################## Book Keeping Functions ####################################
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

################## Prediction and Loss Functions ####################################
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
		x2=biasinv(x_in,FeatMax,FeatMin)
		y_pred=models[k](x2)
		y_pred=bias(y_pred,Emax,Emin)
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
#############################################################
def loss_Force_Energy(Ein,Fin,Etarget,Ftarget,natoms_in_frame,nframes_batch,pF,pE,lVerbose,mode,lossfn):
#############################################################
	#Ein_=torch.sum(Ein,1)
	#for i in range(nframes_batch) :
#		Ein[i]=Ein[i]/natoms_in_frame[i]
	lossE=lossfn(Ein,Etarget)
	lossF=lossfn(Fin,Ftarget)
	ESME=(lossE).item()
	FSME=(lossF).item()
	if(lVerbose) :
		print(mode+' --> ' + 'RMSE Energy : ' +str(torch.sqrt(lossE).item())+' eV/atom')
		print(mode+' --> ' + 'RMSE Force : ' +str(torch.sqrt(lossF).item())+' eV/A')
		CEx=pearsoncorr(Ein,Etarget)
		CFx=pearsoncorr(Fin,Ftarget)
		print(mode+' --> ' + 'CEx : ' +str(CEx.item()))
		print(mode+' --> ' + 'CFx : ' +str(CFx.item()))
	loss=pE*lossE+pF*lossF
	return(loss,ESME,FSME)
#############################################################
def bias(x,Max,Min):
	slope=(Max-Min)/2
	bias=Max-slope
	return(x*slope+bias)
def biasinv(x,Max,Min):
	slope=2/(Max-Min)
	bias=1-Max*slope
	return(x*slope+bias)
#############################################################
def pearsoncorr(output,target) :
#############################################################
# From https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739
	x = output
	y = target

	vx = x - torch.mean(x)
	vy = y - torch.mean(y)

	cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
	return(cost)
