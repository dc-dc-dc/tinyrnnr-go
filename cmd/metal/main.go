package main

import (
	"fmt"

	efficientnetgo "github.com/dc-dc-dc/efficientnet-go"
)

func main() {
	mtl := efficientnetgo.MetalCreateDefaultDevice()
	queue := efficientnetgo.MetalCommandQueue(mtl)
	size := 1024
	a := make([]float32, size)
	for i := range a {
		a[i] = float32(i)
	}
	bufa := efficientnetgo.MetalBuffer(mtl, uint64(len(a)*4))
	b := make([]float32, size)
	efficientnetgo.MetalWriteBuffer(bufa, a)
	fmt.Printf("mtl: %v %v\n", queue, bufa)
	efficientnetgo.MetalReadBuffer(bufa, b)
	kernel := efficientnetgo.MetalLibrary(mtl, `#include <metal_stdlib>\nusing namespace metal;\nkernel void add(device float* a, device float* b, device float* c, uint i [[thread_position_in_grid]]) { c[i] = a[i] + b[i]; }`)
	fmt.Printf("kernel: %v\n", kernel)
}
