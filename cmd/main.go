package main

import (
	"fmt"

	efficientnetgo "github.com/dc-dc-dc/efficientnet-go"
)

func main() {
	platforms := efficientnetgo.GetPlatformIDs()
	if len(platforms) == 0 {
		panic("no OpenCL platform found")
	}

	fmt.Printf("num platforms: %d\n", len(platforms))
	var device efficientnetgo.CLDeviceID
	for _, platform := range platforms {
		devices := efficientnetgo.GetDeviceIDs(platform)
		if len(devices) == 0 {
			panic("no OpenCL device found")
		}
		device = devices[0]
	}
	if device == nil {
		panic("no OpenCL device found")
	}
	ctx := efficientnetgo.CreateContext(device)
	if ctx == nil {
		panic("failed to create OpenCL context")
	}
	cmd_queue := efficientnetgo.CreateCommandQueue(ctx, device)
	if cmd_queue == nil {
		panic("failed to create OpenCL command queue")
	}
	prg := efficientnetgo.CreateProgram(ctx, device, "__kernel void add(__global float *c, __global const float *b, __global const float *a) { int i = get_global_id(0); c[i] = a[i] + b[i]; }")
	if prg == nil {
		panic("failed to create OpenCL program")
	}
	kernel := efficientnetgo.CreateKernel(prg, "add")
	if kernel == nil {
		panic("failed to create OpenCL kernel")
	}

	// Create buffers
	count := 1024
	a := make([]float32, count)
	b := make([]float32, count)
	c := make([]float32, count)
	for i := 0; i < count; i++ {
		a[i] = float32(i)
		b[i] = float32(i)
		c[i] = 0
	}
	bufA := efficientnetgo.CreateBuffer(ctx, cmd_queue, a, false)
	bufB := efficientnetgo.CreateBuffer(ctx, cmd_queue, b, false)
	bufC := efficientnetgo.CreateBuffer(ctx, cmd_queue, c, true)

	efficientnetgo.RunKernel(cmd_queue, kernel, []int64{int64(count)}, []int64{1}, []efficientnetgo.CLMemObject{bufC, bufB, bufA})
	efficientnetgo.ReadBuffer(cmd_queue, bufC, c)
	fmt.Printf("result: %v\n", c)
}
