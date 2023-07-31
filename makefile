.PHONY: run-add run-safe run-opencl opencl

run-add:
	go run ./cmd/add/main.go

run-safe:
	go run ./cmd/safetensors/main.go

run-opencl:
	go run ./cmd/rnnr/main.go

opencl:
	@if [ "OS" = "Darwin" ]; then\
		clang -framework OpenCL extra/opencl.c -o extra/opencl && ./extra/opencl;\
	fi
	@echo $(OS)
	@if [ "$(OS)" = "Windows_NT" ]; then\
		clang -I"${CUDA_PATH}/include" -l"${CUDA_PATH}/lib/x64/OpenCL.lib" extra/opencl.c -o extra/opencl.exe; \
		./extra/opencl.exe;\
	fi