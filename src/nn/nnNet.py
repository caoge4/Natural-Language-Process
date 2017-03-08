# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 18:32:37 2016
network construction

@author: Xie, Jinghui
"""
import cPickle as cp
import numpy as np
import theano.tensor as T
from layerModule import *
import sys
sys.path.append("..")
from util.util import *
from collections import OrderedDict
class model:
	def __init__(self,state):
		self.params = []
		self.model = []
	def saveModel(self,model_path):
		#model_file = open(model_path,'wb')
		dparams=OrderedDict()
		for pp in self.params:
			value = pp.get_value(borrow=True)
			name = pp.name
			dparams[name] = value
			print name
		np.savez(model_path,**dparams)
			
	def loadModel(self,model_path):
		existModel = np.load(model_path)
		for paramKey, value in existModel.iteritems():
			#The dimension of loaded parameter and paramater setting need to be the same 
			self.params[paramKey].set_value(value,borrow=True)

class SLU_RNN_net(model):
	def __init__(self,state):
		self.state = state

		# load the input information
		#word2vec = open(cp.load(state["word2vec"]),'r')
		
		#input tensor definition
		#size:batch*length*wdim word embedding
		#self.batchWordsEmbed = T.ftensor3('batchWordsEmbed')
		
		#size:batch*length*pdim phraselist embedding
		#self.batchPhraseEmbed = T.ftensor3('batchPhraseEmbed')

		#size:batch*length
		self.batchUttInput = T.imatrix('batchUttInput')

		#size:batch*length
		self.batchUttLable = T.imatrix('batchUttLable')

		#LabelMask
		self.mask = T.neq(self.batchUttInput,state["out"]).astype('int8').dimshuffle(1, 0,"x")#.dimshuffle((0, 1, 'x')).repeat(self.state['wembSize'],axis=2)

		#size:
		self.mode = T.iscalar('mode')

		#Word Embeddinf Layer
		WordEmbNet = WordEmbedLayerModule(state)

		#Label Embeddinf Layer
		LabelEmdNet = LabelEmbedLayerModule(state)

		#Get word embedding
		self.batchWordsEmbed = WordEmbNet.getWordEmbedding()[self.batchUttInput]

		#Build encoder	   
		EncoderNet = InputLayerModule(self.state)
		
		#Build decoder
		DecoderNet = OutputLayerModule(self.state)

		#Build softmax layer
		#LrNet = SoftmaxLayerModule(self.state)

		#Based on the type to bulid the encoder
		self.EncoderHiddenState = EncoderNet.build(self.batchWordsEmbed,self.mask)

		#Based on the type to bulid the decoder
		#DecoderOutput:batch*length*labelSize
		self.y = DecoderNet.build(self.EncoderHiddenState,self.batchUttLable,LabelEmdNet.getLabelEmbedding(),self.mode)

		self.params = EncoderNet.getParams() + DecoderNet.getParams() + WordEmbNet.getParams() + LabelEmdNet.getParams() 
		
	def buildTrainFunc(self):
		self.labelEst = T.argmax(self.y,axis=2)
		self.lmask = T.neq(self.batchUttInput,self.state["out"]).astype('int8')
		labelRel = self.batchUttLable * self.lmask
		labelunMatchFlag = T.neq(self.labelEst * self.lmask,labelRel)
		labelunMatchCount = labelunMatchFlag.sum().astype('float32')
		labelTotalCount = self.lmask.sum().astype('float32')
		self.Labelaccu = 1 - labelunMatchCount/labelTotalCount
		uttunMatchFlag = labelunMatchFlag.sum(1)
		uttMatchCount = T.neq(uttunMatchFlag,0).sum().astype('float32')
		uttTotalCount = T.shape(labelunMatchFlag)[0].astype('float32')
		self.Uttaccu = 1 - uttMatchCount/uttTotalCount
		#print self.state["labelSize"]
		self.tmp = theano.tensor.identity_like(np.eye(self.state["labelSize"],dtype="int32"))
		#print type(tmp)
		self.batchlabelReal = self.tmp[self.batchUttLable].astype("float32") 
		self.batchProb = (self.y * self.batchlabelReal).max(axis=2)
		e = 1e-7
		self.clipBatchProb = T.clip(self.batchProb,e,1-e)
		#lmask=T.neq(self.batchUttInput,self.state["out"]).astype('int8').dimshuffle((0, 1, 'x')).repeat(self.state['labelSize'],axis=2)
		
		self.clipBatchProb = self.clipBatchProb * self.lmask.astype("float32")
		self.clipBatchProb = T.extra_ops.compress(self.clipBatchProb>0,self.clipBatchProb)
		self.costValue = -T.log(self.clipBatchProb).sum()
		for pp in self.params:
			self.costValue += self.state["decayRate"] * (pp**2).sum() / self.state['batchSize']
		self.updates = self.calUpdates()
		#build the train function
		self.trainFunc = theano.function(inputs=[self.batchUttInput,self.batchUttLable,self.mode],outputs=[self.Labelaccu,self.Uttaccu,self.costValue,self.labelEst, self.y],updates=self.updates)
		return self.trainFunc
	def calUpdates(self):
		#print self.state
		optType = self.state["optimizer"]
		grads = T.grad(self.costValue,wrt=self.params)
		'''
		grad_clip = 5.
		if grad_clip > 0.:
			g2 = 0.
			for g in grads:
				g2 += (g**2).sum()
			new_grads = []
			for g in grads:
				new_grads.append(T.switch(g2 > (grad_clip**2),g / T.sqrt(g2) * grad_clip,g))
			grads = new_grads
		'''
		#updates=zip(self.params,grads)
		#self.grads = T.as_tensor_variable(grads)
		if optType == "SGD":
			newValue = []
			for idx,param in enumerate(self.params):
				newValue.append(param - self.state["learnRate"] * grads[idx] /  self.state['batchSize'])
		#print "1"#type(grads)#,type(newValue),type(self.params)
		updates = zip(self.params,newValue)
		return updates
		
		
		
		
	def buildTestFunc(self):			   
		#network structure construction
		pass

class CNN_net(model):
	def __init__(self,state):
		pass
