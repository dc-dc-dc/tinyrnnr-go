.PHONY: run-add run-safe opencl

run-add:
	go run ./cmd/add/main.go

run-safe:
	go run ./cmd/safetensors/main.go

opencl:
	clang -framework OpenCL extra/opencl.c -o extra/opencl && ./extra/opencl