# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 00:11:06 2016
For creation of my own Deep learning algorithm based on Theano.

@author: Xie, Jinghui
"""

import argparse
import theano
from nnNet import *
import logging
from dataIter import *
import json

'''
For parsing the argument from the input
'''
def Arguparser():
	parser = argparse.ArgumentParser()  
	parser.add_argument('--config_file', required=True,help='config file for executing the program')  
	args = parser.parse_args()  
	return args
def checkGrads(grad):
	pass
 
 
def main(args):
	state = json.load(open(args.config_file, 'r')) 
	#with open(args.config_file) as f:
	#	for line in f:
	#		params = f.split("=")
	#		if len(params) >1:
	#			state[params[0].lower().strip()] = params[1]

	logging.basicConfig(level=getattr(logging,'DEBUG'),format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

	logging.debug('Start Time')
	
	#Define the epoches and batch size for training 
	#Prepare the data for training
	logging.debug('Build the Data Iterator')
	trainData = BatchData(state)  

	#batch count 
	batchNum = trainData.batchNum
 
	#get data iterator
	DataIterator = trainData.geneDataIter()

	#initialize the network,including the structure and the parameter
	state = trainData.getState()
	net = SLU_RNN_net(state)
	#build the training function
	logging.debug('Build the training function')
	trainFunc = net.buildTrainFunc()
	#build the test function
	######testFunc = net.buildTestFunc() 

	#Start to training the algorithm
	epoch = 0
	while epoch < state['epoch']:
	#perform data generation
  
		batchIdx = 0
		#perform training
		#batchNum = 1
		data,lable = DataIterator.next()
		while batchIdx < batchNum:
			
			#train the dataset
			#print np.shape(data)
			#print lable
			Accu,Accu2,CostValue,yEst,grad = trainFunc(data,lable,1)
			#modelName = "epoch"+str("epoch")+"batch"+str(batchIdx)+".npz"
			#net.saveModel(modelName)
			#print EncoderHiddenState
			#print yEst
			checkGrads(grad)
			#log print 
			logging.debug("For Epoch#{}Steps#{}, the label accrancy:{},the utt accrancy:{}, the cost :{}".format(epoch,batchIdx,Accu,Accu2,CostValue))
			batchIdx = batchIdx + 1		
		epoch = epoch + 1
		if epoch%20 ==0:
			modelName = "epoch"+str(epoch)+".npz"
			net.saveModel(modelName)


		# training has run out, dump the final training model
		
		


if __name__ == '__main__':
	args = Arguparser()
	main(args)
