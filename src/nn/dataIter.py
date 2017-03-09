# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 22:40:27 2016

@author: bio
"""
import math
import cPickle as cp
import os
import json
import numpy as np
class BatchData:
	def __init__(self,state):
		self.state = state
		self.batchSize = state['batchSize']
		self.max_x = 0
		trainSet=[]
		trainLabelSet=[]
		for root,dirs,files in os.walk(state['data']):
			for file in files:
				if 'pkl' in file:
					fileDir = os.path.join(root,file)
					train,val,test,dicts = cp.load(open(fileDir))
					trainSet = trainSet + train[0]
					trainLabelSet = trainLabelSet + train[2]
					self.w2idx, _, self.labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']
					self.idx2w = dict((v,k) for k,v in self.w2idx.items())
					self.idx2la = dict((v,k) for k,v in self.labels2idx.items())
		if state.has_key('dataIndex') and state['dataIndex'] != "":
			index_str = open(state['dataIndex'],'r')
			index = [i for i in index_str]
			index = map(lambda x: int(x),index)
			self.data = trainSet[index]
			self.label = trainLabelSet[index]
		else:
			self.data = trainSet
			self.label = trainLabelSet
		self.dataLen = len(self.data)
		self.max_x = 1 + max(map(lambda x: len(x),self.data))
		self.batchNum =  math.ceil(self.dataLen/self.batchSize)
		self.state['labelSize']=len(self.idx2la)
		self.state['wordSize']=len(self.idx2w)
		#set data mask out flag:
		self.state['out']=-1
		self.state['wordEmdInit'] = self.geneEmdInit()

	def geneEmdInit(self):
		w2vMt = np.zeros([len(self.idx2w),self.state["wembSize"] ],dtype="float32")
		if os.path.exists("w2v.npz"):
			w2vMt = np.load("w2v.npz")
		else:
			with open(self.state["wordEmd"],"r") as w2v:
				for line in w2v:
					thisline = line.split(" ")
					word = thisline[0]
					if self.w2idx.has_key(word):
						vector = [float(i) for i in thisline[1::] ]
						w2vMt[self.w2idx[word]] = np.asarray(vector)
			np.save("w2v.npz",w2vMt)

		return w2vMt
		
				
	def geneDataIter(self):
		batchCnt = 0
		RandomIdx = np.random.permutation(self.dataLen)
		AllData = self.data[RandomIdx]
		Alllabel = self.label[RandomIdx]
		while batchCnt < self.batchNum:
			indexStrt = batchCnt * self.batchSize
			indexEnd = (batchCnt + 1) * self.batchSize - 1
			if indexEnd > self.dataLen-1:
				indexEnd = self.dataLen-1
			batchTrain = [[],[]]
			batchTrainData = AllData[indexStrt:indexEnd+1]
			batchTrainLabel = Alllabel[indexStrt:indexEnd+1]
			batchNormTrainData = np.zeros([len(batchTrainData),self.max_x],dtype="int32") - 1
			batchNormTrainLabel = np.zeros([len(batchTrainData),self.max_x],dtype="int32") + self.labels2idx['O']
			for index,data in enumerate(batchTrainData):
				batchNormTrainData[index,0:len(data)] = data
				batchNormTrainLabel[index,0:len(data)] = batchTrainLabel[index]
			batchTrain[0] = batchNormTrainData
			batchTrain[1] = batchNormTrainLabel
			batchCnt = batchCnt + 1
						
			yield batchTrain
	def generateUtterLabel(self,outputDir):
		fileHandler = open(outputDir,"w")
		for i,uttIdx in enumerate(self.data):
			uttList = map(lambda x: self.idx2w[x], uttIdx)
			utt=""
			for word in uttList:
				utt = utt + word + " "
			uttLabelList = map(lambda x: self.idx2la[x], self.label[i])
			uttLabel=""
			for wordLabel in uttLabelList:
				uttLabel = uttLabel + wordLabel + " "
			thisLine = utt + "\t" + uttLabel + "\n"
			fileHandler.write(thisLine)
	def getState(self):
		return self.state



if __name__=="__main__": 
	config_file = "../../config/cnnConfig.json"
	state = json.load(open(config_file, 'r'))
	batch_data = BatchData(state)
	print "Data size:{},maximum utterance length:{}".format(batch_data.dataLen,batch_data.max_x)
	outputDir = "output.txt"
	batch_data.generateUtterLabel(outputDir)
	dataIterator = batch_data.geneDataIter()
	i = 0
	while i < batch_data.batchNum:
		i = i + 1
		batchData = dataIterator.next()
		print "the {}-th batch size is length{}".format(i,len(batchData[0]))
		print "the {}-th batch size is length{}".format(i,len(batchData[0][0]))
		if i == batch_data.batchNum -1 :
			print batchData[0]
			print batchData[1]	
