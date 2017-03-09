# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:31:23 2016
For each layer construction
@author: bio
"""
import theano.tensor as T
import theano
import numpy as np
import cPickle as cp
import sys
sys.path.append("..")
import sys
from util.util import *
from cell import *

class layerModule:
	def __init__(self,state):
		self.state = state
		self.init_param()
	def init_param(self):
		self.params=[]
		pass
	def getParams(self):
		return self.params
    
class WordEmbedLayerModule(layerModule):
	def __init__(self,state):
		layerModule.__init__(self,state)
		self.init_param()
	def init_param(self): 
		self.params = []
		wordSize = self.state['wordSize']
		EmbSize = self.state['wembSize']
		if self.state.has_key('wordEmd'):
			#print "test"
			self.wordEmbAll = theano.shared(self.state['wordEmdInit'],'wordEmdAll')
		else:
			self.wordEmbAll = theano.shared(random_init((wordSize,EmbSize),rng_type='xavier'),'wordEmdAll')
		self.params.append(self.wordEmbAll)
		
		
	def getWordEmbedding(self):
		return self.wordEmbAll

	def getParams(self):
		return self.params
		
class LabelEmbedLayerModule(layerModule):
	def __init__(self,state):
		layerModule.__init__(self,state)
		self.init_param()
	def init_param(self): 
		self.params = []
		labelSize = self.state['labelSize']
		lEmbSize = self.state['lembSize']
		self.labelEmb = theano.shared(random_init((labelSize,lEmbSize),rng_type='xavier'),'labelEmd')
		self.params.append(self.labelEmb)

	def getLabelEmbedding(self):
		return self.labelEmb
		
	def getParams(self):
		return self.params

class InputLayerModule(layerModule):
	def  __init__(self,state):
		layerModule.__init__(self,state)
		self.init_param()
	def  init_param(self):
		inputDim = self.state["wembSize"]
		outputDim = self.state["hiddenSize"]	
		self.En_W_z = theano.shared(random_init((inputDim,outputDim),rng_type="normal"),'En_W_z')
		self.En_U_z = theano.shared(random_init((outputDim,outputDim),rng_type="normal"),'En_U_z')
		self.En_b_z = theano.shared(np.zeros((outputDim,), dtype=theano.config.floatX),'En_b_z')
		self.En_W_r = theano.shared(random_init((inputDim,outputDim),rng_type="normal"),'En_W_r')
		self.En_U_r = theano.shared(random_init((outputDim,outputDim),rng_type="normal"),'En_U_r')
		self.En_b_r = theano.shared(np.zeros((outputDim,), dtype=theano.config.floatX),'En_b_r')
		self.En_W_h = theano.shared(random_init((inputDim,outputDim),rng_type="normal"),'En_W_h')
		self.En_U_h = theano.shared(random_init((outputDim,outputDim),rng_type="normal"),'En_U_h')
		self.En_b_h = theano.shared(np.zeros((outputDim,), dtype=theano.config.floatX),'En_b_h')
		self.params=[self.En_W_z, self.En_U_z, self.En_b_z,self.En_W_r, self.En_U_r, self.En_b_r,self.En_W_h, self.En_U_h, self.En_b_h]

		#self.gruCell.getParams() #+ [self.h0]

	def gated_step_func(self,input_x,mask,h_tm1):
		#update gate
		z_t = T.nnet.sigmoid(T.dot(input_x,self.En_W_z) + T.dot(h_tm1,self.En_U_z) + self.En_b_z)
		#reset gate
		r_t = T.nnet.sigmoid(T.dot(input_x,self.En_W_r) + T.dot(h_tm1,self.En_U_r) + self.En_b_r)
		#memory update
		h_t_f = T.tanh(T.dot(input_x,self.En_W_h) + r_t * T.dot(h_tm1,self.En_U_h) + self.En_b_h)
		#hidden state
		h_t = ( np.float32(1) - z_t) * h_t_f + z_t * h_tm1
		
		mask = mask.repeat(self.state['hiddenSize'],axis=1)
		h_t = mask*h_t + (np.float32(1)-mask)*h_tm1 
		return h_t
	def build(self,batchWordsVec,mask):
        #input batch * maxlength * wdim
		batchWordsVec = batchWordsVec.dimshuffle(1,0,2)
		
		#batchH0 = T.alloc(self.Deh0,batchWordsVec.shape[1])		
		batchH0 = T.alloc(np.float32(0),batchWordsVec.shape[1],self.state["hiddenSize"])
		hiddenState,_ = theano.scan(self.gated_step_func,sequences=[batchWordsVec,mask], outputs_info=batchH0)
		return hiddenState
        
	def getParams(self):
		return self.params
        
class OutputLayerModule(layerModule):
	def  __init__(self,state):
		layerModule.__init__(self,state)
		self.init_param()
	def  init_param(self): 
		outputDim = self.state["hiddenSize"]
		inputDim = self.state["hiddenSize"]
		lDim = self.state['lembSize']
		labelSize = self.state['labelSize']
		deInputDim = inputDim+lDim
		#(labelSize+inputDim) * HiddenDim
		self.De_W_z = theano.shared(random_init((deInputDim,outputDim),rng_type="normal"),'De_W_z')
		self.De_U_z = theano.shared(random_init((outputDim,outputDim),rng_type="normal"),'De_U_z')
		self.De_b_z = theano.shared(np.zeros((outputDim,), dtype=theano.config.floatX),'De_b_z')
		self.De_W_r = theano.shared(random_init((deInputDim,outputDim),rng_type="normal"),'De_W_r')
		self.De_U_r = theano.shared(random_init((outputDim,outputDim),rng_type="normal"),'De_U_r')
		self.De_b_r = theano.shared(np.zeros((outputDim,), dtype=theano.config.floatX),'De_b_r')
		self.De_W_h = theano.shared(random_init((deInputDim,outputDim),rng_type="normal"),'De_W_h')
		self.De_U_h = theano.shared(random_init((outputDim,outputDim),rng_type="normal"),'De_U_h')
		self.De_b_h = theano.shared(np.zeros((outputDim,), dtype=theano.config.floatX),'De_b_h')
		self.params=[self.De_W_z, self.De_U_z, self.De_b_z,self.De_W_r, self.De_U_r, self.De_b_r,self.De_W_h, self.De_U_h, self.De_b_h]

		#self.Deh0 = theano.shared(name='Deh0'+suffix,value=np.zeros((outputDim,),dtype=theano.config.floatX))
		#self.Dey0 = theano.shared(name='Dey0'+suffix,value=np.zeros((lDim,),dtype=theano.config.floatX))
		#self.Del0 = theano.shared(name='Del0'+suffix,value=np.zeros((labelSize,),dtype=theano.config.floatX))
		#self.params = self.gruCell.getParams() #+ [self.Deh0]
		#labelSize * HiddenDim
		#self.SoftmaxW0 = theano.shared(np.zeros([outputDim,labelSize],dtype=theano.config.floatX),'SoftmaxW0')
		self.SoftmaxW0 = theano.shared(random_init((outputDim,labelSize),rng_type="uniform"),'SoftmaxW0')
		#labelSize
		self.SoftmaxB0 = theano.shared(np.zeros((labelSize,),dtype=theano.config.floatX),'SoftmaxW0')
		#self.SoftmaxB0 = theano.shared(np.zeros((labelSize,),dtype=theano.config.floatX),'SoftmaxW0')
		self.params += [self.SoftmaxW0,self.SoftmaxB0]

	def gated_step_func(self,xt,lt,h_tm1,lemdm1,lm1,L_emd,mode):
		#h_t = self.gruCell.build(hm1,T.concatenate([xt,lemdm1],axis=1))
		input_x = T.concatenate([xt,lemdm1],axis=1)
		#input_x = xt
		#update gate
		z_t = T.nnet.sigmoid(T.dot(input_x,self.De_W_z) + T.dot(h_tm1,self.De_U_z) + self.De_b_z)
		#reset gate
		r_t = T.nnet.sigmoid(T.dot(input_x,self.De_W_r) + T.dot(h_tm1,self.De_U_r) + self.De_b_r)
		#memory update
		h_t_f = T.tanh(T.dot(input_x,self.De_W_h) + r_t * T.dot(h_tm1,self.De_U_h) + self.De_b_h)
		#hidden state
		h_t = (np.float32(1)-z_t) * h_t_f + z_t * h_tm1
		
		lt_pre = T.nnet.softmax(T.dot(h_t,self.SoftmaxW0) + self.SoftmaxB0)
		lemd_t_est = L_emd[lt_pre.argmax(axis=1)]
		#output the hard coded label embedding 
		lemd_t = T.switch(T.eq(mode,1),L_emd[lt],lemd_t_est)
		return h_t,lemd_t,lt_pre

	def build(self,hiddenStateVec,batchL,L_emd,mode): 
		#input1 maxlength * batch * hiddenDim
		#input2 batch * maxlength
		#batchH0 = T.alloc(self.Deh0,batchWordsVec.shape[1])
		batchL = batchL.dimshuffle(1,0)
		batchH0 = T.alloc(np.float32(0),np.shape(hiddenStateVec)[1],self.state["hiddenSize"])		
		batchY0 = T.alloc(np.float32(0),np.shape(hiddenStateVec)[1],self.state["lembSize"])
		batchL0 = T.alloc(np.float32(0),np.shape(hiddenStateVec)[1],self.state["labelSize"])
		_res,_ = theano.scan(self.gated_step_func, sequences= [hiddenStateVec,batchL], outputs_info=[batchH0,batchY0,batchL0],non_sequences=[L_emd,mode])
		decoderOutput = _res[2].dimshuffle(1,0,2)
		return decoderOutput

	def getParams(self):
		return self.params

class SoftmaxLayerModule(layerModule):
	def  __init__(self,state):
		layerModule.__init__(self,state)
		self.init_param()
	def  init_param(self): 
		pass
	def  build(self,decoderInput):
		#Input:maxlength * batch * hiddenDim
		#Output:batch * maxlength * lDim
		softOutput = T.nnet.softmax(T.dot(decoderInput,self.SoftmaxW0) + self.SoftmaxB0)
		return softOutput

class dropoutLayer(layerModule):		
	def  __init__(self,state):
		layerModule.__init__()
		init_param()
	def  build(self,dropoutInput):
		dropoutOutput = dropoutInput
		return dropoutOutput

class batchnormLayer(layerModule):	
	def  __init__(self,state):
		layerModule.__init__()
		init_param()
	def  build(self,batchnormInput):
		batchnormOutput = batchnormInput
		return batchnormOutput

