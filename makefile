.PHONY: run-add run-safe run-opencl opencl

run-add:
	go run ./cmd/add/main.go

run-safe:
	go run ./cmd/safetensors/main.go

run-opencl:
	go run ./cmd/efficientnet/main.go

opencl:
	clang -framework OpenCL extra/opencl.c -o extra/opencl && ./extra/opencl