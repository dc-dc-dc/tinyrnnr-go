package tinyrnnrgo

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation
#import <Metal/Metal.h>

void * _MTLCreateSystemDefaultDevice() {
	return (void *)MTLCreateSystemDefaultDevice();
}

void * _MTLCommandQueue(void *device) {
	return (void *)[(id)device newCommandQueue];
}

void * _MTLBuffer(void *device, uint64 size) {
	return (void *)[(id)device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

void * _MTLLibrary(void *device, const char *src) {
	NSString *source = [NSString stringWithUTF8String:src];
	MTLCompileOptions* compileOptions = [MTLCompileOptions new];
	return (void *)[(id)device newLibraryWithSource:source options:compileOptions error:nil];
}


*/
import "C"
import "unsafe"

func MetalCreateDefaultDevice() uintptr {
	return uintptr(C._MTLCreateSystemDefaultDevice())
}

func MetalCommandQueue(device uintptr) uintptr {
	return uintptr(C._MTLCommandQueue(unsafe.Pointer(device)))
}

func MetalLibrary(device uintptr, src string) uintptr {
	c_src := C.CString(src)
	defer C.free(unsafe.Pointer(c_src))
	return uintptr(C._MTLLibrary(unsafe.Pointer(device), c_src))
}

func MetalBuffer(device uintptr, size uint64) uintptr {
	return uintptr(C._MTLBuffer(unsafe.Pointer(device), C.uint64(size)))
}

func MetalWriteBuffer(buf uintptr, in []float32) error {
	copy(unsafe.Slice((*float32)(unsafe.Pointer(buf)), len(in)), in)
	return nil
}

func MetalReadBuffer(buf uintptr, out []float32) error {
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(buf)), len(out)))
	return nil
}
