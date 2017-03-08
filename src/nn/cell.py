import theano.tensor as T
import sys
sys.path.append("..")
import sys
from util.util import *
import numpy as np
class cellModule:
	def __init__(self,state):
		self.state = state
		self.init_param()
	def init_param(self,suffix=""):
		self.params=[]
		pass

class gru(cellModule):
	def __init__(self,state,inputDim,outputDim,activeType="tanh",suffix=""):
		#cellModule.__init__(self,state)
		self.state = state
		self.inputDim = inputDim		
		self.outputDim = outputDim		
		self.init_param()
	def init_param(self,suffix=""):
		self.W_z = theano.shared(random_init((self.inputDim,self.outputDim),rng_type="xavier"),'W_z'+suffix)
		self.U_z = theano.shared(random_init((self.outputDim,self.outputDim),rng_type="xavier"),'U_z'+suffix)
		self.b_z = theano.shared(np.zeros((self.outputDim,), dtype=theano.config.floatX),'b_z'+suffix)
		self.W_r = theano.shared(random_init((self.inputDim,self.outputDim),rng_type="xavier"),'W_r'+suffix)
		self.U_r = theano.shared(random_init((self.outputDim,self.outputDim),rng_type="xavier"),'U_r'+suffix)
		self.b_r = theano.shared(np.zeros((self.outputDim,), dtype=theano.config.floatX),'b_r'+suffix)
		self.W_h = theano.shared(random_init((self.inputDim,self.outputDim),rng_type="xavier"),'W_h'+suffix)
		self.U_h = theano.shared(random_init((self.outputDim,self.outputDim),rng_type="xavier"),'U_h'+suffix)
		self.b_h = theano.shared(np.zeros((self.outputDim,), dtype=theano.config.floatX),'b_h'+suffix)
		self.params=[self.W_z, self.U_z, self.b_z,self.W_r, self.U_r, self.b_r,self.W_h, self.U_h, self.b_h]
	
	def build(self,h_tm1,input_x):
		#update gate
		z_t = T.nnet.sigmoid(T.dot(input_x,self.W_z) + T.dot(h_tm1,self.U_z) + self.b_z)
		#reset gate
		r_t = T.nnet.sigmoid(T.dot(input_x,self.W_r) + T.dot(h_tm1,self.U_r) + self.b_r)
		#memory update
		h_t_f = T.tanh(T.dot(input_x,self.W_h) + r_t * T.dot(h_tm1,self.U_h) + self.b_h)
		#hidden state
		h_t = z_t * h_t_f + ( 1 - z_t) * h_tm1
		return h_t
	def getParams(self):
		return self.params
		
