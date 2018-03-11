.PHONY: build
build: ## build the code
	nvcc -std=c++11 -O3 -use_fast_math --expt-extended-lambda --expt-relaxed-constexpr -I../ -I/usr/local/cuda/include main.cu -lcurand

.DEFAULT_GOAL := build
