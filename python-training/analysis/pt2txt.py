import numpy as np
import torch
from torch import nn
frc_predict_train=torch.load('../data05/train-frc-pred.0000000199.pt')
frc_ground_train=torch.load('../data01/train-frc-ground-truth.pt')
#print(frc_predict_train)
nfrc=frc_predict_train.size()[0]
frc_file=open('forces_train.txt','w')
print(frc_ground_train.size())
print(frc_predict_train.size())
#frc_ground_train=torch.tensor(frc_ground_train,dtype=torch.float64)
#frc_predict_train=torch.tensor(frc_predict_train,dtype=torch.float64)
for i in range(nfrc) :
	frc_file.write( "%.6E %.6E\n" %(frc_predict_train[i].item(), frc_ground_train[i].item()))
lossfun=nn.MSELoss()
loss=lossfun(frc_predict_train,frc_ground_train)
print(torch.sqrt(loss))
eng_predict_train=torch.load('../data05/train-eng-pred.0000000199.pt')
eng_ground_train=torch.load('../data01/train-eng-ground-truth.pt')
#print(eng_predict_train)
neng=eng_predict_train.size()[0]
eng_file=open('eng_train.txt','w')
for i in range(neng) :
	eng_file.write( "%.6E %.6E\n" %(eng_predict_train[i].item(), eng_ground_train[i].item()))
