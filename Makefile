SHELL = /bin/bash

push-predict-train:
	@cp predict_train.py predict.py &&cog push r8.im/douwantech/gpt-sovits-train

push-predict-generate:
	@cp predict_generate.py predict.py &&cog push r8.im/douwantech/gpt-sovits-generate
