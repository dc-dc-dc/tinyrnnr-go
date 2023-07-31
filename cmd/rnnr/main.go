package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"os"

	"golang.org/x/image/draw"

	tinyrnnrgo "github.com/dc-dc-dc/tinyrnnr-go"
)

var (
	modelFile  *string
	tensorFile *string
)

func init() {
	modelFile = flag.String("model", "net.json", "tinygrad model file")
	tensorFile = flag.String("tensor", "net.safetensors", "tinygrad safetensors file")
}

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
	fmt.Printf("len: %f %f %f %f\n", input[0], input[1], input[2], input[3])
	return input
}

func main() {
	flag.Parse()
	fmt.Printf("running model: %s tensors: %s\n", *modelFile, *tensorFile)

	labels := []string{}
	fd, err := os.Open("imagenet-simple-labels.json")
	if err != nil {
		panic(err)
	}
	defer fd.Close()
	if err := json.NewDecoder(fd).Decode(&labels); err != nil {
		panic(err)
	}

	model, err := tinyrnnrgo.NewModelFromFile(*modelFile)
	if err != nil {
		panic(err)
	}
	tensor, err := tinyrnnrgo.NewSafeTensorFromFile(*tensorFile)
	if err != nil {
		panic(err)
	}
	if err := model.Setup(tensor); err != nil {
		panic(err)
	}

	// load test image
	img, err := os.Open("hen.jpg")
	if err != nil {
		panic(err)
	}
	defer img.Close()

	src, err := jpeg.Decode(img)
	if err != nil {
		panic(err)
	}

	input := prepareImage(model.GetInputSize()/4, src)
	tfd, err := os.Create("temp")
	if err != nil {
		panic(err)
	}
	defer tfd.Close()
	for i := range input {
		fmt.Fprintf(tfd, "%f,", input[i])
	}
	out, err := model.Run(input)
	if err != nil {
		panic(err)
	}
	var index int
	fmt.Printf("len: %f %f %f\n", out[0], out[1], out[2])
	for i := range out {
		if out[i] > out[index] && out[i] != 0 {
			// fmt.Printf("new max %d %s %.5f\n", i, labels[i], out[i])
			index = i
		}
	}
	fmt.Printf("result: %s\n", labels[index])
}
