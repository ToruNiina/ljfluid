.PHONY: help
help: ## print this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## build the code
	g++ -Wall -Wextra -Wpedantic -std=c++14 -O3 -ffast-math -march=native -I../ main.cpp -o main.out

.PHONY: run
run: ## run simulation and write the result to "traj.xyz" and "energy.dat"
	time ./main.out 2>&1 1>traj.xyz | tee energy.dat

.PHONY: clean
clean: ## remove executable binary, "traj.xyz" and "energy.dat".
	rm traj.xyz energy.dat main.out

.PHONY: rerun
rerun: clean bluid run ## re-build and re-run simulation.

.DEFAULT_GOAL := help
