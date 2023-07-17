package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"math"
	"os"

	efficientnetgo "github.com/dc-dc-dc/efficientnet-go"
)

func populateSize(size []uint64) []uint64 {
	for len(size) < 3 {
		size = append(size, 1)
	}
	return size
}

func convertToFloatArray(arr []uint8) []float32 {
	res := make([]float32, len(arr)/4)
	for i := 0; i < len(arr); i += 4 {
		bits := binary.LittleEndian.Uint32(arr[i : i+4])
		res[i/4] = math.Float32frombits(bits)
	}
	return res
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

	model, err := efficientnetgo.NewModelFromFile("net.json")
	if err != nil {
		panic(err)
	}
	tensor, err := efficientnetgo.NewSafeTensorFromFile("net.safetensors")
	if err != nil {
		panic(err)
	}
	platforms := efficientnetgo.GetPlatformIDs()
	if len(platforms) == 0 {
		panic("no platforms")
	}
	platform := platforms[0]
	devices := efficientnetgo.GetDeviceIDs(platform)
	if len(devices) == 0 {
		panic("no devices")
	}
	device := devices[0]
	ctx := efficientnetgo.CreateContext(device)
	if ctx == nil {
		panic("no context")
	}
	queue := efficientnetgo.CreateCommandQueue(ctx, device)
	if queue == nil {
		panic("no queue")
	}
	// Create the bufs and kernels
	bufs := map[string]efficientnetgo.CLMemObject{}
	for key, buf := range model.Buffers {
		temp := efficientnetgo.CreateBuffer(ctx, queue, buf.Size, buf.Name == "")
		if buf.Name != "" {
			efficientnetgo.WriteBuffer(queue, temp, tensor.GetTensor(buf.Name))
		}
		bufs[key] = temp
	}
	bufs["input"] = efficientnetgo.CreateBuffer(ctx, queue, model.InputSize, false)
	bufs["outputs"] = efficientnetgo.CreateBuffer(ctx, queue, model.OutputSize, true)
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
	rgba := image.NewRGBA(src.Bounds())
	draw.Draw(rgba, src.Bounds(), src, src.Bounds().Min, draw.Src)

	input := make([]float32, model.InputSize)
	temp := convertToFloatArray(rgba.Pix)
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			tx := (x / 224) * src.Bounds().Dx()
			ty := (y / 224) * src.Bounds().Dy()
			for c := 0; c < 3; c++ {
				input[c*224*224+y*224+x] = (temp[ty*src.Bounds().Dx()*3+tx*3+c]/255.0 - 0.45) / 0.225
			}
		}
	}
	efficientnetgo.WriteBuffer(queue, bufs["input"], input)
	// Create the kernels
	kernels := map[string]efficientnetgo.CLKernel{}
	for key, kernel := range model.Functions {
		prg := efficientnetgo.CreateProgram(ctx, device, kernel)
		if prg == nil {
			panic("error creating kernel")
		}
		kernels[key] = efficientnetgo.CreateKernel(prg, key)
	}
	// Run the kerenels in order
	for _, statement := range model.Statements {
		args := make([]efficientnetgo.CLMemObject, len(statement.Args))
		for i, arg := range statement.Args {
			args[i] = bufs[arg]
		}
		statement.LocalSize = populateSize(statement.LocalSize)
		statement.GlobalSize = populateSize(statement.GlobalSize)
		for i, v := range statement.LocalSize {
			statement.GlobalSize[i] = statement.GlobalSize[i] * v
		}
		fmt.Printf("running kernel %s global: %v local: %v\n", statement.Kernel, statement.GlobalSize, statement.LocalSize)
		if res := efficientnetgo.RunKernel(queue, kernels[statement.Kernel], statement.GlobalSize, statement.LocalSize, args); res != "" {
			panic(res)
		}
	}
	var index int
	out := make([]float32, model.OutputSize)
	efficientnetgo.ReadBuffer(queue, bufs["outputs"], out)
	fmt.Printf("got output: %v\n", out)
	for i := range out {
		if out[i] > out[index] {
			fmt.Printf("new max: %f\n", out[i])
			index = i
		}
	}
	fmt.Printf("result: %s\n", labels[index])
}
