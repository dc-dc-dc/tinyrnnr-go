package efficientnetgo

import (
	"encoding/json"
	"os"
)

type Statement struct {
	Kernel     string   `json:"kernel"`
	Args       []string `json:"args"`
	GlobalSize []uint64 `json:"global_size"`
	LocalSize  []uint64 `json:"local_size"`
}

type Buffer struct {
	Size uint64 `json:"size"`
	Name string `json:"id"`
}

type ModelMetadata struct {
	Backend    string            `json:"backend"`
	InputSize  uint64            `json:"input_size"`
	OutputSize uint64            `json:"output_size"`
	Functions  map[string]string `json:"functions"`
	Statements []Statement       `json:"statements"`
	Buffers    map[string]Buffer `json:"buffers"`
}

func NewModelFromFile(path string) (*ModelMetadata, error) {
	fd, err := os.Open("net.json")
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	model := &ModelMetadata{}
	if err := json.NewDecoder(fd).Decode(model); err != nil {
		return nil, err
	}
	return model, nil
}
