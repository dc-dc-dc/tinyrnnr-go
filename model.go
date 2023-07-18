package tinyrnnrgo

import (
	"encoding/json"
	"fmt"
	"os"
)

type Statement struct {
	Kernel     string   `json:"kernel"`
	Args       []string `json:"args"`
	GlobalSize []uint64 `json:"global_size"`
	LocalSize  []uint64 `json:"local_size"`
}

type BufferMetadata struct {
	Size  uint64 `json:"size"`
	Dtype string `json:"dtype"`
	Name  string `json:"id"`
}

type ModelMetadata struct {
	Backend    string                    `json:"backend"`
	InputSize  BufferMetadata            `json:"input_size"`
	OutputSize BufferMetadata            `json:"output_size"`
	Functions  map[string]string         `json:"functions"`
	Statements []Statement               `json:"statements"`
	Buffers    map[string]BufferMetadata `json:"buffers"`
}

type Backend interface {
	Setup() error
	CreateKernel(name, kernel string) (Kernel, error)
	RunKernel(kernel Kernel, global, local []uint64, bufs []Buffer) error
	CreateBuffer(size uint64, writable bool) (Buffer, error)
	ReadBuffer(buf Buffer, out []float32) error
	WriteBuffer(buf Buffer, in []float32) error
}

type Kernel interface{}
type Buffer interface{}
type Model struct {
	metadata *ModelMetadata
	backend  Backend
	kernels  map[string]Kernel
	buffers  map[string]Buffer
}

func NewModelFromFile(path string) (*Model, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	metadata := &ModelMetadata{}
	if err := json.NewDecoder(fd).Decode(metadata); err != nil {
		return nil, err
	}
	if metadata.Backend != "GPU" {
		return nil, fmt.Errorf("unsupported backend %s", metadata.Backend)
	}
	return NewModel(metadata), nil
}

func NewModel(metadata *ModelMetadata) *Model {
	return &Model{
		backend:  NewOpenCLBackend(),
		metadata: metadata,
		kernels:  map[string]Kernel{},
		buffers:  map[string]Buffer{},
	}
}

func (m *Model) Setup(tensor *SafeTensor) error {
	if err := m.backend.Setup(); err != nil {
		return err
	}
	// Create the bufs
	for key, buf := range m.metadata.Buffers {
		temp, err := m.backend.CreateBuffer(buf.Size*4, buf.Name == "")
		if err != nil {
			return fmt.Errorf("error creating buffer %s: %v", key, err)
		}
		if buf.Name != "" {
			if err := m.backend.WriteBuffer(temp, tensor.GetTensor(buf.Name)); err != nil {
				return fmt.Errorf("error writing buffer %s: %v", key, err)
			}
		}
		m.buffers[key] = temp
	}
	input, err := m.backend.CreateBuffer(m.metadata.InputSize.Size*4, false)
	if err != nil {
		return fmt.Errorf("error creating input buffer: %v", err)
	}
	m.buffers["input"] = input
	output, err := m.backend.CreateBuffer(m.metadata.OutputSize.Size*4, true)
	if err != nil {
		return fmt.Errorf("error creating output buffer: %v", err)
	}
	m.buffers["outputs"] = output
	// Create the kernels
	for key, kernel := range m.metadata.Functions {
		kernel, err := m.backend.CreateKernel(key, kernel)
		if err != nil {
			return fmt.Errorf("error creating kernel %s: %v", key, err)
		}
		m.kernels[key] = kernel
	}
	return nil
}

func (m *Model) GetInputSize() uint64 {
	return m.metadata.InputSize.Size
}

func (m *Model) Run(input []float32) ([]float32, error) {
	if err := m.backend.WriteBuffer(m.buffers["input"], input); err != nil {
		return nil, err
	}
	for _, statement := range m.metadata.Statements {
		args := make([]Buffer, len(statement.Args))
		for i, arg := range statement.Args {
			args[i] = m.buffers[arg]
		}
		statement.LocalSize = populateSize(statement.LocalSize)
		statement.GlobalSize = populateSize(statement.GlobalSize)
		for i, v := range statement.LocalSize {
			statement.GlobalSize[i] = statement.GlobalSize[i] * v
		}

		// fmt.Printf("running kernel %s global: %v local: %v\n", statement.Kernel, statement.GlobalSize, statement.LocalSize)
		if res := m.backend.RunKernel(m.kernels[statement.Kernel], statement.GlobalSize, statement.LocalSize, args); res != nil {
			return nil, res
		}
	}
	out := make([]float32, m.metadata.OutputSize.Size)
	if err := m.backend.ReadBuffer(m.buffers["outputs"], out); err != nil {
		return nil, err
	}
	return out, nil
}

func populateSize(size []uint64) []uint64 {
	for len(size) < 3 {
		size = append(size, 1)
	}
	return size
}
