.PHONY: run opencl

run:
	go run ./cmd/main.go

opencl:
	clang -framework OpenCL extra/opencl.c -o extra/opencl && ./extra/opencl