.PHONY: build
build: ## build the code
	nvcc -std=c++11 --expt-extended-lambda -I../ -I/usr/local/cuda/include main.cu -lcurand

.DEFAULT_GOAL := build
