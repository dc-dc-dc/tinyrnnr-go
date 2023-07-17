package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"os"

	"golang.org/x/image/draw"

	efficientnetgo "github.com/dc-dc-dc/efficientnet-go"
)

func convertToFloatArray(arr []uint8) []float32 {
	res := make([]float32, len(arr))
	for i := 0; i < len(arr); i++ {
		res[i] = float32(arr[i])
	}
	return res
}

func prepareImage(size uint64, src image.Image) []float32 {
	rgba := image.NewRGBA(image.Rect(0, 0, 224, 224))
	draw.NearestNeighbor.Scale(rgba, rgba.Bounds(), src, src.Bounds(), draw.Over, nil)
	input := make([]float32, size)
	temp := convertToFloatArray(rgba.Pix)
	for i := range temp {
		temp[i] = (temp[i]/255.0)*.45 - .225
	}
	i := 0
	for c := 0; c < 3; c++ {
		for x := 0; x < 224*224; x++ {
			input[i] = temp[x*4+c]
			i += 1
		}
	}
	return input
}

func main() {
	labels := []string{}
	fd, err := os.Open("imagenet-simple-labels.json")
	if err != nil {
		panic(err)
	}
	defer fd.Close()
	if err := json.NewDecoder(fd).Decode(&labels); err != nil {
		panic(err)
	}

	model, err := efficientnetgo.NewModelFromFile("net-opencl.json")
	if err != nil {
		panic(err)
	}
	tensor, err := efficientnetgo.NewSafeTensorFromFile("net.safetensors")
	if err != nil {
		panic(err)
	}
	if err := model.Setup(tensor); err != nil {
		panic(err)
	}

	// load test image
	img, err := os.Open("Norwegian_hen.jpg")
	if err != nil {
		panic(err)
	}
	defer img.Close()

	src, err := jpeg.Decode(img)
	if err != nil {
		panic(err)
	}

	input := prepareImage(model.GetInputSize(), src)
	out, err := model.Run(input)
	if err != nil {
		panic(err)
	}
	var index int
	for i := range out {
		if out[i] > out[index] {
			index = i
		}
	}
	fmt.Printf("result: %s\n", labels[index])
}
