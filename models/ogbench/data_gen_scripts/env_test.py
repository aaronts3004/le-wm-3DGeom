import ctypes
import os

libdir = os.path.join(
    os.environ["VIRTUAL_ENV"],
    "lib/python3.10/site-packages/nvidia/cu13/lib"
)

ctypes.CDLL(os.path.join(libdir, "libnvrtc.so.13"))
print("libnvrtc loaded")

ctypes.CDLL(os.path.join(libdir, "libnvrtc-builtins.so.13.0"))
print("builtins loaded")