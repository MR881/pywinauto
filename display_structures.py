import ctypes
from ctypes import wintypes

# Define the callback function type (MonitorEnumProc)
MONITORENUMPROC = ctypes.WINFUNCTYPE(
    ctypes.c_int,
    wintypes.HMONITOR,  # handle to display monitor
    wintypes.HDC,       # handle to monitor DC
    wintypes.LPRECT,    # pointer to monitor intersection rectangle (This is the RECT Structure)
    wintypes.LPARAM     # data
)

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', ctypes.c_uint32),
        ('biWidth', ctypes.c_long),
        ('biHeight', ctypes.c_long),
        ('biPlanes', ctypes.c_short),
        ('biBitCount', ctypes.c_short),
        ('biCompression', ctypes.c_uint32),
        ('biSizeImage', ctypes.c_uint32),
        ('biXPelsPerMeter', ctypes.c_long),
        ('biYPelsPerMeter', ctypes.c_long),
        ('biClrUsed', ctypes.c_uint32),
        ('biClrImportant', ctypes.c_uint32)
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ('bmiHeader', BITMAPINFOHEADER),
        ('bmiColors', ctypes.c_uint32 * 1)  # This can be adjusted based on the number of colors used
    ]
