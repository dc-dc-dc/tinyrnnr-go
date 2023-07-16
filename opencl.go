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

type MemObject = C.cl_mem

func GetPlatformIDs() []C.cl_platform_id {
	numPlatforms := C.cl_uint(0)
	if C.clGetPlatformIDs(0, nil, &numPlatforms) != C.CL_SUCCESS {
		return nil
	}
	if numPlatforms == 0 {
		return nil
	}
	platforms := make([]C.cl_platform_id, numPlatforms)
	if C.clGetPlatformIDs(numPlatforms, &platforms[0], nil) != C.CL_SUCCESS {
		return nil
	}
	return platforms
}

func GetDeviceIDs(platform_id C.cl_platform_id) []C.cl_device_id {
	numDevices := C.cl_uint(0)
	if C.clGetDeviceIDs(platform_id, C.CL_DEVICE_TYPE_ALL, 0, nil, &numDevices) != C.CL_SUCCESS {
		return nil
	}
	if numDevices == 0 {
		return nil
	}
	devices := make([]C.cl_device_id, numDevices)
	if C.clGetDeviceIDs(platform_id, C.CL_DEVICE_TYPE_ALL, numDevices, &devices[0], nil) != C.CL_SUCCESS {
		return nil
	}
	return devices
}

func CreateContext(device_id C.cl_device_id) C.cl_context {
	var err C.cl_int
	context := C.clCreateContext(nil, 1, &device_id, nil, nil, &err)
	if err != C.CL_SUCCESS {
		return nil
	}
	return context
}

func CreateCommandQueue(ctx C.cl_context, device C.cl_device_id) C.cl_command_queue {
	var err C.cl_int
	queue := C.clCreateCommandQueue(ctx, device, 0, &err)
	if err != C.CL_SUCCESS {
		return nil
	}
	return queue
}

func CreateProgram(ctx C.cl_context, device C.cl_device_id, src string) C.cl_program {
	c_src := C.CString(src)
	defer C.free(unsafe.Pointer(c_src))
	var err C.cl_int
	program := C.clCreateProgramWithSource(ctx, 1, &c_src, nil, &err)
	if err != C.CL_SUCCESS {
		return nil
	}
	if C.clBuildProgram(program, 1, &device, nil, nil, nil) != C.CL_SUCCESS {
		return nil
	}
	return program
}

func CreateKernel(program C.cl_program, name string) C.cl_kernel {
	c_name := C.CString(name)
	defer C.free(unsafe.Pointer(c_name))
	var err C.cl_int
	kernel := C.clCreateKernel(program, c_name, &err)
	if err != C.CL_SUCCESS {
		return nil
	}
	return kernel
}

func getBufferType(write bool) C.ulonglong {
	if write {
		return C.CL_MEM_WRITE_ONLY
	}
	return C.CL_MEM_READ_ONLY
}

func CreateBuffer(ctx C.cl_context, queue C.cl_command_queue, buf []float32, write bool) MemObject {
	var err C.cl_int
	size := C.size_t(len(buf) * 4)

	buffer := C.clCreateBuffer(ctx, getBufferType(write), size, nil, &err)

	if err != C.CL_SUCCESS {
		return nil
	}
	if !write {
		if C.clEnqueueWriteBuffer(queue, buffer, C.CL_TRUE, 0, size, unsafe.Pointer(&buf[0]), 0, nil, nil) != C.CL_SUCCESS {
			return nil
		}
	}
	return buffer
}

func ReadBuffer(queue C.cl_command_queue, buffer C.cl_mem, buf []float32) {
	size := C.size_t(len(buf) * 4)
	if C.clEnqueueReadBuffer(queue, buffer, C.CL_TRUE, 0, size, unsafe.Pointer(&buf[0]), 0, nil, nil) != C.CL_SUCCESS {
		return
	}
}

func RunKernel(queue C.cl_command_queue, kernel C.cl_kernel, global_size, local_size []int64, bufs []MemObject) {
	for i, buf := range bufs {
		if C.clSetKernelArg(kernel, C.cl_uint(i), C.size_t(unsafe.Sizeof(buf)), unsafe.Pointer(&buf)) != C.CL_SUCCESS {
			return
		}
	}
	if C.clEnqueueNDRangeKernel(queue, kernel, C.cl_uint(len(global_size)), nil, (*C.size_t)(unsafe.Pointer(&global_size[0])), (*C.size_t)(unsafe.Pointer(&local_size[0])), 0, nil, nil) != C.CL_SUCCESS {
		return
	}
	return
}
