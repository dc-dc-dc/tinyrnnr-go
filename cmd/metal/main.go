package main

import (
	"fmt"

	tinyrnnrgo "github.com/dc-dc-dc/tinyrnnr-go"
)

func main() {
	mtl := tinyrnnrgo.MetalCreateDefaultDevice()
	queue := tinyrnnrgo.MetalCommandQueue(mtl)
	size := 1024
	a := make([]float32, size)
	for i := range a {
		a[i] = float32(i)
	}
	bufa := tinyrnnrgo.MetalBuffer(mtl, uint64(len(a)*4))
	b := make([]float32, size)
	tinyrnnrgo.MetalWriteBuffer(bufa, a)
	fmt.Printf("mtl: %v %v\n", queue, bufa)
	tinyrnnrgo.MetalReadBuffer(bufa, b)
	kernel := tinyrnnrgo.MetalLibrary(mtl, `#include <metal_stdlib>\nusing namespace metal;\nkernel void add(device float* a, device float* b, device float* c, uint i [[thread_position_in_grid]]) { c[i] = a[i] + b[i]; }`)
	fmt.Printf("kernel: %v\n", kernel)
}
