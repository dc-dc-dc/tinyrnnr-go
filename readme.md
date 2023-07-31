# Tinnyrnnr GO

A simple model runner companion for [tinygrad](https://github.com/tinygrad/tinygrad)

## Usage
    
1. Export your model from tinygrad
```bash
GPU=1 python examples/compile_efficientnet.py
```

2. Run the model with tinnyrnnr
```bash
go run ./cmd/rnnr/main.go -model <path/to/tinygrad>/tinygrad/examples/net.json -tensor <path/to/tinygrad>/tinygrad/examples/net.safetensors
```