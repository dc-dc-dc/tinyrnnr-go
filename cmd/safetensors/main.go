package main

import (
	"fmt"

	tinyrnnrgo "github.com/dc-dc-dc/tinyrnnr-go"
)

func main() {
	safetensor, err := tinyrnnrgo.NewSafeTensorFromFile("./net.safetensors")
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", safetensor.GetTensor("_conv_stem")[0:10])
}
