.PHONY: run opencl

run-add:
	go run ./cmd/add/main.go

opencl:
	clang -framework OpenCL extra/opencl.c -o extra/opencl && ./extra/opencl