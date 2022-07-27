import numpy as np
class read_xsf  :
	def __init__(self,path_2_coords) :
		self.isghost=False
		read_file=open(path_2_coords,'r')
		print('opened : ' +path_2_coords)
		line=read_file.readline()
		line=line.strip().split()
		#print(line)
		self.Energy=float(line[-1])
		self.stress=np.zeros(6)
		print('Total Energy ' + str(self.Energy)+' eV' )
		while True : 
			line=read_file.readline()
			if not line :
				break
			#print(line)
			fl=line.strip().split()[0]
			if(fl=='PRIMVEC') :
				#print(line)
				self.Hmat=np.zeros((3,3))
				for j in range(3) :
					line=read_file.readline()
					line=line.strip().split()
					self.Hmat[j,0]=float(line[0])
					self.Hmat[j,1]=float(line[1])
					self.Hmat[j,2]=float(line[2])
				#print(Hmat)

			if(fl=='PRESSURE') :
				#print(line)
				line=read_file.readline()
				line=line.strip().split()
				for i in range(6) :
					self.stress[i]=float(line[i])
			if(fl=='PRIMCOORD') :
				line=read_file.readline()
				line=line.strip().split()
				self.natom=int(line[0])
				self.pos=np.zeros((self.natom,3))
				self.recip=np.zeros((self.natom,3))
				self.forces=np.zeros((self.natom,3))
				self.types=list()
				for j in range(self.natom) :
					line=read_file.readline()
					line=line.strip().split()
					self.types.append(line[0])
					for icart in range(3) :
						self.pos[j,icart]=float(line[icart+1])
						self.forces[j,icart]=float(line[icart+4])
					#print(self.Hmat)
					#print(np.linalg.inv(self.Hmat))
					#print(self.positions[j,:])
					self.recip[j,:]=np.matmul(np.linalg.inv(self.Hmat),self.pos[j,:])
		read_file.close()
	def getghosted(self,RC=7.5,NBUFFER=None) :
	# Create Buffer or Ghost Cells
		if NBUFFER is None : 
			self.NBUFFER=self.natom*27
		else :
			self.NBUFFER=NBUFFER
		self.isghost=True
	#	self.NBUFFER=self.natom*27
		self.RC=RC
		#Lmax=np.max(self.Hmat)
		rmin=1.0*np.array([-RC/self.Hmat[0,0],-RC/self.Hmat[1,1],-RC/self.Hmat[2,2]])
		rmax=1.0*np.array([self.Hmat[0,0]+RC/self.Hmat[0,0],
			self.Hmat[1,1]+RC/self.Hmat[1,1],
			self.Hmat[2,2]+RC/self.Hmat[2,2]])
	#	rmin=1.0*np.array([-RC/Lmax,-RC/Lmax/RC,-RC/Lmax])
		#rmax=1.0*np.array([self.Hmat[0,0]+RC/Lmax,
		#	self.Hmat[1,1]+RC/Lmax,
	#		self.Hmat[2,2]+RC/Lmax])
		recip=self.recip.copy()
		self.recip=np.zeros((self.NBUFFER,3))
		start=0
		end=self.natom
		self.recip[start:end,:]=recip[:,:]
		#print(self.recip[0:self.natom,:])
		self.cptr=end
		#print(self.recip[self.cptr,:])
		self.cptr=end-1
		#print(self.recip[self.cptr,:])
		i=0
		for nx in [-1,0,1] :
			for ny in [-1,0,1] :
				for nz in [-1,0,1] :
					if(nx==0 and ny==0 and nz==0) :
						continue
					#print(i)
					add=self.recip[0:self.natom,:]+np.array([nx,ny,nz]) 
					for j in range(self.natom) :
						r=add[j,:]
						if(r[0]<rmin[0] or r[0]>rmax[0]) :
							continue
						elif(r[1]<rmin[1] or r[1]>rmax[1]) :
							continue
						elif(r[2]<rmin[2] or r[2]>rmax[2]) :
							continue
						else :
							self.cptr=self.cptr+1
							self.recip[self.cptr,:]=r
						
						
					#print(add.shape)
					#print((start,end))
					#print(self.NBUFFER)
					#addreal=np.matmul(self.Hmat,add)
					#self.recip[start:end,:]=add[:,:]
					#start=start+self.natom
					#end=end+self.natom
					#self.positions=np.vstack(self.positions,addreal)
					i=i+1
		self.pos=np.zeros((self.NBUFFER,3))
		for i in range(self.NBUFFER) :
			self.pos[i,:]=np.matmul(self.Hmat,self.recip[i,:])
			
	def get_nbr_list_ON2(self,MAXNEIGHBORS) :
		self.MAXNEIGBORS=MAXNEIGHBORS
		self.nbrlist=np.zeros((self.natom,self.MAXNEIGBORS),dtype=np.int32)
		if not self.isghost :
			raise('err : need to make ghost cells for nbrlist')
		for i in range(self.natom) :
			for j in range(self.cptr +1) :
				if(i==j) : 
					continue 
				rij=np.sqrt(np.sum((self.pos[i,:]-self.pos[j,:])**2))
				if(rij<self.RC) :
					self.nbrlist[i,0]=self.nbrlist[i,0]+1
					self.nbrlist[i,self.nbrlist[i,0]]=j
	def get_nbr_list_ON2_pbc(self,MAXNEIGHBORS,RC) :
		self.RC=RC
		self.MAXNEIGBORS=MAXNEIGHBORS
		self.nbrlist=np.zeros((self.natom,self.MAXNEIGBORS),dtype=np.int32)
	#	if not self.isghost :
	#		raise('err : need to make ghost cells for nbrlist')
		for i in range(self.natom) :
			for j in range(self.natom) :
				if(i==j) : 
					continue 
				dr=self.recip[i,:]-self.recip[j,:]
				for k in range(3) :
					if(dr[k]>0.5) :
						dr[k]=dr[k]-1.0
					elif(dr[k]< -0.5) :
						dr[k]=dr[k]+1.0
				dr=np.matmul(self.Hmat,dr)
					
				rij=np.sqrt(np.sum((dr)**2))
				if(rij<self.RC) :
					self.nbrlist[i,0]=self.nbrlist[i,0]+1
					self.nbrlist[i,self.nbrlist[i,0]]=j
	def linkedlist_pbc(self) :
		self.layer_shift=0
		self.cc=np.zeros(3,dtype=np.int32)
		lcsize=np.zeros(3)
		LX=self.Hmat[0,0]
		LY=self.Hmat[1,1]
		LZ=self.Hmat[2,2]
		self.cc[0]=np.floor(LX/self.RC)
		self.cc[1]=np.floor(LY/self.RC)
		self.cc[2]=np.floor(LZ/self.RC)
		#print(cc[0])
		#print(cc[1])
		#print(cc[2])
		lcsize[0]=LX/self.cc[0]
		lcsize[1]=LY/self.cc[1]
		lcsize[2]=LZ/self.cc[2]
		l1=[]
		l2=[]
		l3=[]
		MAXLAYERS=10
		self.NBUFFER=self.natom
		self.llist=np.zeros(self.NBUFFER,dtype=np.int32)
		self.header=np.ones((MAXLAYERS,MAXLAYERS,MAXLAYERS),dtype=np.int32)*-1
		self.nacell=np.zeros((MAXLAYERS,MAXLAYERS,MAXLAYERS),dtype=np.int32)
		for n in range(self.natom) :
			#print(n)
			l =np.int32(np.floor(self.pos[n,:]/lcsize[:])+self.layer_shift)
			l1.append(l[0])
			l2.append(l[1])
			l3.append(l[2])
			#print(l)
			self.llist[n] = self.header[l[0], l[1], l[2]]
			self.header[l[0], l[1], l[2]] = n
			self.nacell[l[0], l[1], l[2]] = self.nacell[l[0], l[1], l[2]] + 1
		#print('L stat')
		#print(max(l1))
		#print(min(l1))
		#print(max(l2))
		#print(min(l2))
		#print(max(l3))
		#print(min(l3))
		#print('L stat')

	def get_nbr_list_pbc(self,MAXNEIGHBORS,RC) :
		self.RC=RC
		self.MAXNEIGBORS=MAXNEIGHBORS
		self.nbrlist=np.zeros((self.natom,self.MAXNEIGBORS),dtype=np.int32)
		tot=0
	#	if not self.isghost :
	#		raise('err : need to make ghost cells for nbrlist')
		for c1 in range(0,self.cc[0]) :
			for c2 in range(0,self.cc[1]) :
				for c3 in range(0,self.cc[2]) :
					c1=c1
					c2=c2
					c3=c3
					i = self.header[c1, c2, c3]
					#print(i)
					print('c1 c2 c3 '+str(c1)+' '+str(c2)+ ' '+str(c3))
					tot=tot+self.nacell[c1, c2, c3]
					#print(self.nacell[c1, c2, c3])
					for i1 in range(self.nacell[c1, c2, c3]) :
						for c4 in [-1,0,1] :
							for c5 in [-1,0,1] :
								for c6 in [-1,0,1] :
									ic= [c1+c4, c2+c5, c3+c6]
									ic[0]=ic[0]%self.cc[0]
									ic[1]=ic[1]%self.cc[1]
									ic[2]=ic[2]%self.cc[2]

									#print('c1 c2 c3 '+str(c1)+' '+str(c2)+ ' '+str(c3))
									#print('ic1 ic2 ic3 '
								#		+ str(ic[0])+' '+str(ic[1])+ ' '+str(ic[2]))
									#print('c4 c5 c6 '
								 #		+ str(c4)+' '+str(c5)+ ' '+str(c6))
									#print(ic)
									j = self.header[ic[0],ic[1],ic[2]]
	#								print('i ,j ' + str(i) + ' ' + str(j))
									for j1 in range(self.nacell[ic[0],ic[1],ic[2]]) :
										if(i!=j) :
											dr=self.recip[i,:]-self.recip[j,:]
											for k in range(3) :
												if(dr[k]>0.5) :
													dr[k]=dr[k]-1.0
												elif(dr[k]< -0.5) :
													dr[k]=dr[k]+1.0
											dr=np.matmul(self.Hmat,dr)
												
											rij=np.sqrt(np.dot(dr,dr))
											#print(rij)
											if(rij<self.RC) :
												self.nbrlist[i,0]=(
													self.nbrlist[i,0]+1)
												ine=self.nbrlist[i,0]
												#print(ine)
												print('i ,j ' + 
													str(i) + ' ' + str(j))
												print(ine)
												self.nbrlist[i,ine]=j
										j=self.llist[j]
						i=self.llist[i]
		#print('tot')
		#print(tot)
	def __del__(self):
		pass

'''
frame=read_xsf('den_3_0110.xsf')
#print(frame.Hmat)
#print(frame.pos[0,:])
print(frame.types)
print(set(frame.types))
print(list(set(frame.types)))
test=(list(set(frame.types)))
print(test[0])
#print(frame.forces[0,:])
#print(frame.recip[0,:])
#frame.getghosted()
#print(frame.cptr)
#print(frame.pos[frame.cptr-1 , : ])
#print(frame.pos[frame.cptr+1 , : ])
frame.get_nbr_list_ON2_pbc(MAXNEIGHBORS=400,RC=7.5)
print(frame.nbrlist[0,:])
#print(frame.nbrlist[50,:])
print(frame.nbrlist[319,:])
frame.linkedlist_pbc()
#frame.linkedlist()
#print(frame.llist)
#print(frame.cc)
frame.get_nbr_list_pbc(MAXNEIGHBORS=400,RC=7.5)
print(frame.nbrlist[0,:])
#print(frame.nbrlist[50,:])
print(frame.nbrlist[319,:])
#print(frame.recip.shape)
#print(frame.positions.shape)
#print(frame.NBUFFER)
'''
