import ctypes
import sys

lib_name = "libcudart.so.12"
print(f"Attempting to load {lib_name}...")

try:
    cudart = ctypes.CDLL(lib_name)
    print(f"Successfully loaded {lib_name}")
except OSError as e:
    print(f"Failed to load {lib_name}: {e}")
    sys.exit(1)

count = ctypes.c_int()
try:
    ret = cudart.cudaGetDeviceCount(ctypes.byref(count))
    print(f"cudaGetDeviceCount returned: {ret}")
    if ret == 0:
        print(f"GPU Count: {count.value}")
    else:
        print(f"cudaGetDeviceCount failed with error code: {ret}")
except Exception as e:
    print(f"Error calling cudaGetDeviceCount: {e}")
