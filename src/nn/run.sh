THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python RnnSRL.py --config_file ../../config/SluRnnConfig.json
#THEANO_FLAGS=mode=FAST_COMPILE,device=cpu,exception_verbosity=high,floatX=float32 python RnnClassifier.py --config_file ../../config/RnnConfig.json
#THEANO_FLAGS=mode=DEBUG_MODE,device=cpu,floatX=float32 python RnnClassifier.py --config_file ../../config/RnnConfig.json
