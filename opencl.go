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
import "unsafe"

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

func CreateBuffer(ctx CLContext, queue CLCommandQueue, buf []float32, write bool) CLMemObject {
	var err CLint
	size := C.size_t(len(buf) * 4)

	buffer := C.clCreateBuffer(ctx, getBufferType(write), size, nil, &err)

	if err != CL_SUCCESS {
		return nil
	}
	if !write {
		if C.clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, unsafe.Pointer(&buf[0]), 0, nil, nil) != CL_SUCCESS {
			return nil
		}
	}
	return buffer
}

func ReadBuffer(queue CLCommandQueue, buffer CLMemObject, buf []float32) {
	size := C.size_t(len(buf) * 4)
	if C.clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, unsafe.Pointer(&buf[0]), 0, nil, nil) != CL_SUCCESS {
		return
	}
}

func RunKernel(queue CLCommandQueue, kernel CLKernel, global_size, local_size []int64, bufs []CLMemObject) {
	for i, buf := range bufs {
		if C.clSetKernelArg(kernel, C.cl_uint(i), C.size_t(unsafe.Sizeof(buf)), unsafe.Pointer(&buf)) != CL_SUCCESS {
			return
		}
	}
	if C.clEnqueueNDRangeKernel(queue, kernel, C.cl_uint(len(global_size)), nil, (*C.size_t)(unsafe.Pointer(&global_size[0])), (*C.size_t)(unsafe.Pointer(&local_size[0])), 0, nil, nil) != CL_SUCCESS {
		return
	}
	return
}
