SHELL = /bin/bash

push-predict-train:
	@sudo rm -rf results/*
	@sudo rm -rf logs/*
	@sudo rm -rf input/*
	@sudo rm -rf output/*
	@sudo rm -rf GPT_weights/*
	@sudo rm -rf SoVITS_weights/*
	@cp predict_train.py predict.py &&cog push r8.im/douwantech/gpt-sovits-train

push-predict-generate:
	@cp predict_generate.py predict.py &&cog push r8.im/douwantech/gpt-sovits-generate