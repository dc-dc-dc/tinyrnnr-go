package main

import (
	"fmt"

	efficientnetgo "github.com/dc-dc-dc/efficientnet-go"
)

func main() {
	safetensor, err := efficientnetgo.NewSafeTensorFromFile("./net.safetensors")
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", safetensor.GetTensor("_conv_stem")[0:10])
}
