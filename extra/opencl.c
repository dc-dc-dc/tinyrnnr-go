#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>

#define HANDLE(f)                     \
    {                                 \
        if (f != CL_SUCCESS)          \
        {                             \
            printf("Error: %d\n", f); \
            return 1;                 \
        }                             \
    }

// clang -framework OpenCL extra/opencl.c -o extra/opencl
int main(int argc, char *argv[])
{
    cl_uint platform_count;
    HANDLE(clGetPlatformIDs(0, NULL, &platform_count));
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
    HANDLE(clGetPlatformIDs(platform_count, platforms, NULL));
    printf("Platform count: %d\n", platform_count);
    cl_device_id *devices = NULL;
    for (int i = 0; i < platform_count; i++)
    {
        cl_uint device_count;
        HANDLE(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count));
        printf("Device count %d\n", device_count);
        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * device_count);
        HANDLE(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices, NULL));
        // printf("Platform %d:\n", i);
        // char *name;
        // size_t name_size;
        // printf("ID: %d\n", platforms[i]);
        // HANDLE(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &name_size));
        // name = (char *)malloc(sizeof(char) * name_size);
        // HANDLE(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, name, NULL));
        // printf("Name: %s\n", name);
    }
    if (devices == NULL)
    {
        printf("no devices found\n");
        return -1;
    }
    cl_int error_code;
    cl_context clctx = clCreateContext(NULL, 1, devices, NULL, NULL, &error_code);
    if (error_code != CL_SUCCESS)
    {
        printf("Error creating context: %d\n", error_code);
        return -1;
    }
    cl_command_queue clcmdq = clCreateCommandQueue(clctx, devices[0], 0, &error_code);
    if (error_code != CL_SUCCESS)
    {
        printf("Error creating context: %d\n", error_code);
        return -1;
    }
    const char *src = "__kernel void add(__global float *c, __global const float *b, __global const float *a) { int i = get_global_id(0); c[i] = a[i] + b[i]; }";
    cl_program clprog = clCreateProgramWithSource(clctx, 1, &src, NULL, &error_code);
    if (error_code != CL_SUCCESS)
    {
        printf("Error creating program: %d\n", error_code);
        return -1;
    }

    HANDLE(clBuildProgram(clprog, 1, &devices[0], NULL, NULL, NULL));
    cl_kernel kernel = clCreateKernel(clprog, "add", &error_code);
    if (error_code != CL_SUCCESS)
    {
        printf("Error creating kernel: %d\n", error_code);
        return -1;
    }
    const int N = 1024;
    float *a = (float *)malloc(sizeof(float) * N);
    float *b = (float *)malloc(sizeof(float) * N);
    float *c = (float *)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }
    cl_mem cl_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY, sizeof(float) * N, NULL, &error_code);
    if (error_code != CL_SUCCESS)
    {
        printf("Error creating buffer: %d\n", error_code);
        return -1;
    }
    cl_mem cl_b = clCreateBuffer(clctx, CL_MEM_READ_ONLY, sizeof(float) * N, NULL, &error_code);
    if (error_code != CL_SUCCESS)
    {
        printf("Error creating buffer: %d\n", error_code);
        return -1;
    }
    cl_mem cl_c = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, &error_code);
    if (error_code != CL_SUCCESS)
    {
        printf("Error creating buffer: %d\n", error_code);
        return -1;
    }
    HANDLE(clEnqueueWriteBuffer(clcmdq, cl_a, CL_TRUE, 0, sizeof(float) * N, a, 0, NULL, NULL));
    HANDLE(clEnqueueWriteBuffer(clcmdq, cl_b, CL_TRUE, 0, sizeof(float) * N, b, 0, NULL, NULL));
    HANDLE(clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_c));
    HANDLE(clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_b));
    HANDLE(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_a));
    size_t global_work_size[1] = {N};
    HANDLE(clEnqueueNDRangeKernel(clcmdq, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL));
    HANDLE(clEnqueueReadBuffer(clcmdq, cl_c, CL_TRUE, 0, sizeof(float) * N, c, 0, NULL, NULL));
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", c[i]);
    }
    return 0;
}