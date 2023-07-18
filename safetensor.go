package tinyrnnrgo

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
)

type TensorMetadata struct {
	DType       string    `json:"dtype"`
	Shape       []uint64  `json:"shape"`
	DataOffsets [2]uint64 `json:"data_offsets"`
}

type SafeTensor struct {
	Metadata map[string]TensorMetadata
	raw      []byte
}

func NewSafeTensorFromFile(path string) (*SafeTensor, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	raw, err := io.ReadAll(fd)
	if err != nil {
		return nil, err
	}
	return NewSafeTensor(raw)
}

func NewSafeTensor(raw []byte) (*SafeTensor, error) {
	n := binary.LittleEndian.Uint64(raw[:8])
	meta := map[string]TensorMetadata{}
	if err := json.Unmarshal(raw[8:n+8], &meta); err != nil {
		return nil, err
	}
	return &SafeTensor{
		Metadata: meta,
		raw:      raw[n+8:],
	}, nil
}

func (st *SafeTensor) GetTensor(name string) []float32 {
	// TODO: This sucks, should just do it on load
	if meta, ok := st.Metadata[name]; ok {
		arr := st.raw[meta.DataOffsets[0]:meta.DataOffsets[1]]
		res := make([]float32, len(arr)/4)
		for i := 0; i < len(arr); i += 4 {
			bits := binary.LittleEndian.Uint32(arr[i : i+4])
			res[i/4] = math.Float32frombits(bits)
		}
		return res
	}
	return nil
}
