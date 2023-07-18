package main

import (
	"fmt"

	tinyrnnrgo "github.com/dc-dc-dc/tinyrnnr-go"
)

func main() {
	cl := tinyrnnrgo.NewOpenCLBackend()
	if err := cl.Setup(); err != nil {
		panic(err)
	}
	kernel, err := cl.CreateKernel("add", "__kernel void add(__global float *c, __global const float *b, __global const float *a) { int i = get_global_id(0); c[i] = a[i] + b[i]; }")
	if err != nil {
		panic(err)
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
	bufA, err := cl.CreateBuffer(uint64(len(a)*4), false)
	if err != nil {
		panic(err)
	}

	bufB, err := cl.CreateBuffer(uint64(len(b)*4), false)
	if err != nil {
		panic(err)
	}
	bufC, err := cl.CreateBuffer(uint64(len(c)*4), true)
	if err != nil {
		panic(err)
	}
	if err := cl.WriteBuffer(bufA, a); err != nil {
		panic(err)
	}
	if err := cl.WriteBuffer(bufB, b); err != nil {
		panic(err)
	}

	if err := cl.RunKernel(kernel, []uint64{uint64(count)}, []uint64{1}, []tinyrnnrgo.Buffer{bufC, bufB, bufA}); err != nil {
		panic(err)
	}
	if err := cl.ReadBuffer(bufC, c); err != nil {
		panic(err)
	}
	fmt.Printf("result: %v\n", c)
}
