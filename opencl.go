package efficientnetgo

/*
#cgo darwin LDFLAGS: -framework OpenCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// status codes
const CL_SUCCESS = 0
const CL_FALSE = 0
const CL_TRUE = 1

const CL_MEM_READ_WRITE = (1 << 0)
const CL_MEM_WRITE_ONLY = (1 << 1)
const CL_MEM_READ_ONLY = (1 << 2)

// common structs
type CLint = C.cl_int
type CLMemObject = C.cl_mem
type CLKernel = C.cl_kernel
type CLProgram = C.cl_program
type CLCommandQueue = C.cl_command_queue
type CLContext = C.cl_context
type CLPlatformID = C.cl_platform_id
type CLDeviceID = C.cl_device_id

type OpenCLBackend struct {
	ctx    CLContext
	queue  CLCommandQueue
	device CLDeviceID
}

type OpenCLBuffer struct {
	buf CLMemObject
}

type OpenCLKernel struct {
	kernel CLKernel
}

func NewOpenCLBackend() Backend {
	return &OpenCLBackend{}
}

func NewOpenCLBuffer(buf CLMemObject) Buffer {
	return &OpenCLBuffer{
		buf: buf,
	}
}

func NewOpenCLKernel(kernel CLKernel) Kernel {
	return &OpenCLKernel{
		kernel: kernel,
	}
}

func (bc *OpenCLBackend) Setup() error {
	platforms := GetPlatformIDs()
	if len(platforms) == 0 {
		return fmt.Errorf("no platforms")
	}
	platform := platforms[0]
	devices := GetDeviceIDs(platform)
	if len(devices) == 0 {
		return fmt.Errorf("no devices")
	}
	bc.device = devices[0]
	bc.ctx = CreateContext(bc.device)
	if bc.ctx == nil {
		return fmt.Errorf("no context")
	}
	bc.queue = CreateCommandQueue(bc.ctx, bc.device)
	if bc.queue == nil {
		return fmt.Errorf("no queue")
	}

	return nil
}

func (bc *OpenCLBackend) CreateKernel(name, src string) (Kernel, error) {
	prg := CreateProgram(bc.ctx, bc.device, src)
	if prg == nil {
		return nil, fmt.Errorf("error creating kernel")
	}
	kernel := CreateKernel(prg, name)
	if kernel == nil {
		return nil, fmt.Errorf("error creating kernel")
	}
	return NewOpenCLKernel(kernel), nil
}

func (bc *OpenCLBackend) CreateBuffer(size uint64, writable bool) (Buffer, error) {
	buf := CreateBuffer(bc.ctx, bc.queue, size, writable)
	if buf == nil {
		return nil, fmt.Errorf("failed to create buffer")
	}
	return NewOpenCLBuffer(buf), nil
}

func (bc *OpenCLBackend) ReadBuffer(buf Buffer, out []float32) error {
	clBuf, ok := buf.(*OpenCLBuffer)
	if !ok {
		return fmt.Errorf("not an opencl buffer")
	}
	return ReadBuffer(bc.queue, clBuf.buf, out)
}
func (bc *OpenCLBackend) WriteBuffer(buf Buffer, in []float32) error {
	clBuf, ok := buf.(*OpenCLBuffer)
	if !ok {
		return fmt.Errorf("not an opencl buffer")
	}
	return WriteBuffer(bc.queue, clBuf.buf, in)
}

func (bc *OpenCLBackend) RunKernel(kernel Kernel, global, local []uint64, bufs []Buffer) error {
	clKernel, ok := kernel.(*OpenCLKernel)
	if !ok {
		return fmt.Errorf("not an opencl kernel")
	}
	clbufs := make([]OpenCLBuffer, len(bufs))
	for i, buf := range bufs {
		clBuf, ok := buf.(*OpenCLBuffer)
		if !ok {
			return fmt.Errorf("buf at %d not an opencl buffer", i)
		}
		clbufs[i] = *clBuf
	}
	return RunKernel(bc.queue, clKernel.kernel, global, local, clbufs)
}

func GetPlatformIDs() []CLPlatformID {
	numPlatforms := C.cl_uint(0)
	if C.clGetPlatformIDs(0, nil, &numPlatforms) != CL_SUCCESS {
		return nil
	}
	if numPlatforms == 0 {
		return nil
	}
	platforms := make([]CLPlatformID, numPlatforms)
	if C.clGetPlatformIDs(numPlatforms, &platforms[0], nil) != CL_SUCCESS {
		return nil
	}
	return platforms
}

func GetDeviceIDs(platform_id CLPlatformID) []CLDeviceID {
	numDevices := C.cl_uint(0)
	if C.clGetDeviceIDs(platform_id, C.CL_DEVICE_TYPE_ALL, 0, nil, &numDevices) != CL_SUCCESS {
		return nil
	}
	if numDevices == 0 {
		return nil
	}
	devices := make([]CLDeviceID, numDevices)
	if C.clGetDeviceIDs(platform_id, C.CL_DEVICE_TYPE_ALL, numDevices, &devices[0], nil) != CL_SUCCESS {
		return nil
	}
	return devices
}

func CreateContext(device_id CLDeviceID) CLContext {
	var err CLint
	context := C.clCreateContext(nil, 1, &device_id, nil, nil, &err)
	if err != CL_SUCCESS {
		return nil
	}
	return context
}

func CreateCommandQueue(ctx CLContext, device CLDeviceID) CLCommandQueue {
	var err CLint
	queue := C.clCreateCommandQueue(ctx, device, 0, &err)
	if err != CL_SUCCESS {
		return nil
	}
	return queue
}

func CreateProgram(ctx CLContext, device CLDeviceID, src string) CLProgram {
	c_src := C.CString(src)
	defer C.free(unsafe.Pointer(c_src))
	var err CLint
	program := C.clCreateProgramWithSource(ctx, 1, &c_src, nil, &err)
	if err != CL_SUCCESS {
		return nil
	}
	if C.clBuildProgram(program, 1, &device, nil, nil, nil) != CL_SUCCESS {
		return nil
	}
	return program
}

func CreateKernel(program CLProgram, name string) CLKernel {
	c_name := C.CString(name)
	defer C.free(unsafe.Pointer(c_name))
	var err CLint
	kernel := C.clCreateKernel(program, c_name, &err)
	if err != CL_SUCCESS {
		return nil
	}
	return kernel
}

func getBufferType(write bool) C.ulonglong {
	if write {
		return CL_MEM_WRITE_ONLY
	}
	return CL_MEM_READ_ONLY
}

func CreateBuffer(ctx CLContext, queue CLCommandQueue, size uint64, write bool) CLMemObject {
	var err CLint
	buffer := C.clCreateBuffer(ctx, getBufferType(write), C.size_t(size), nil, &err)

	if err != CL_SUCCESS {
		return nil
	}
	return buffer
}

func WriteBuffer(queue CLCommandQueue, buffer CLMemObject, buf []float32) error {
	size := C.size_t(len(buf) * 4)
	if status := C.clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, unsafe.Pointer(&buf[0]), 0, nil, nil); status != CL_SUCCESS {
		return fmt.Errorf("failed to write to buffer got status: %d", status)
	}
	return nil
}

func ReadBuffer(queue CLCommandQueue, buffer CLMemObject, buf []float32) error {
	size := C.size_t(len(buf) * 4)
	if status := C.clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, unsafe.Pointer(&buf[0]), 0, nil, nil); status != CL_SUCCESS {
		return fmt.Errorf("failed to read buffer got status: %d", status)
	}
	return nil
}

func RunKernel(queue CLCommandQueue, kernel CLKernel, global_size, local_size []uint64, bufs []OpenCLBuffer) error {
	for i, buf := range bufs {
		if status := C.clSetKernelArg(kernel, C.cl_uint(i), C.size_t(unsafe.Sizeof(buf.buf)), unsafe.Pointer(&buf.buf)); status != CL_SUCCESS {
			return fmt.Errorf("error setting kernel arg %d got status: %d", i, status)
		}
	}
	if status := C.clEnqueueNDRangeKernel(queue, kernel, C.cl_uint(len(global_size)), nil, (*C.size_t)(unsafe.Pointer(&global_size[0])), (*C.size_t)(unsafe.Pointer(&local_size[0])), 0, nil, nil); status != CL_SUCCESS {
		return fmt.Errorf("error enqueeing kernel got status: %d", status)
	}
	return nil
}
