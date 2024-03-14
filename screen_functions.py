import win32con
import win32api
import win32gui
import numpy as np
import cv2
import ctypes

gdi32 = ctypes.windll.gdi32

def capture_display(screen_numbers, region = None):
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    monitors_info = win32api.EnumDisplayMonitors()
    captured_images = []

    for screen_number in screen_numbers:
        if screen_number >= len(monitors_info):
            print(f"Invalid screen number: {screen_number}")
            continue

        if region is None:
            left, top, right, bottom = monitors_info[screen_number][2]

        else:
            x1, y1 = region[0]
            x2, y2 = region[1]
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)

            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)

        width = right - left
        height = bottom - top


        h_desktop_dc = win32gui.GetWindowDC(0)
        h_capture_dc = win32gui.CreateCompatibleDC(h_desktop_dc)
        h_capture_bmp = win32gui.CreateCompatibleBitmap(h_desktop_dc, width, height)
        
        win32gui.SelectObject(h_capture_dc, h_capture_bmp)
        win32gui.BitBlt(h_capture_dc, 0, 0, width, height, h_desktop_dc, left, top, win32con.SRCCOPY)
        

        # Since pywin32 doesn't have the GetDIBits (which is a great function they should really have it)
        # The function has to make an instance of bitmap info and info header for the pointers to work
        # properly...

        ############## Declairing the class structures
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32),
                ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long),
                ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [
                ('bmiHeader', BITMAPINFOHEADER),
                ('bmiColors', ctypes.c_uint32 * 1)
            ]
        ##############
        image_data = ctypes.create_string_buffer(width * height * 4)

        bmp_info = BITMAPINFO()
        bmp_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmp_info.bmiHeader.biWidth = width
        bmp_info.bmiHeader.biHeight = -height  # Negative to indicate a top-down DIB
        bmp_info.bmiHeader.biPlanes = 1
        bmp_info.bmiHeader.biBitCount = 32
        bmp_info.bmiHeader.biCompression = 0


        # This is the properway to grab the winapi handle from the Pyhandle without detaching
        # The .__init__() method is only to be used to create a new instance. But if you're
        # doing that you might as well just call the HANDLE function.
        h_capture_bmp_handle = int(h_capture_bmp)

        # Pywin32 currently doesn't have the GetDIBits function for Bitmap data...
        # https://github.com/mhammond/pywin32/issues/2120
        # So need to call it from gdi32 instead
        gdi32.GetDIBits(h_capture_dc, h_capture_bmp_handle, 0, height, image_data, ctypes.byref(bmp_info), 0)


        win32gui.DeleteObject(h_capture_bmp_handle)
        win32gui.DeleteObject(h_capture_bmp)
        win32gui.DeleteDC(h_capture_dc)
        win32gui.ReleaseDC(0, h_desktop_dc)

        # Convert to an image (NumPy array)
        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
        captured_images.append(image)

    if len(captured_images) > 1:
        return np.hstack(captured_images)
    elif len(captured_images) == 1:
        return captured_images[0]
    else:
        return None

def screenshots(screen_numbers=[0], combined=False, region = None):
    captured_images = []
    for screen_number in screen_numbers:
        screenshot = capture_display(screen_number, region)

    if combined and len(captured_images) > 1:
        combined_screenshot = np.hstack(captured_images)
        cv2.imshow('Combined Screens', combined_screenshot)
    else:
        for i, image in enumerate(captured_images):
            cv2.imshow(f'Monitor {screen_numbers[i] + 1}', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def screen_recorder(screen_numbers=[0], frame_rate=30, combined=False):
        
    if combined:
        cv2.namedWindow('Combined Screens', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Combined Screens', 1280, 480)  # Adjust size as needed
    else:
        for screen_number in screen_numbers:
            cv2.namedWindow(f'Monitor {screen_number + 1}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Monitor {screen_number + 1}', 640, 480)

    while True:
        captured_images = []
        for screen_number in screen_numbers:
            screenshot = capture_display([screen_number])
            if screenshot is not None and screenshot.shape[0] > 0 and screenshot.shape[1] > 0:
                captured_images.append(screenshot)
            else:
                print(f"Invalid screenshot dimensions for Screen {screen_number + 1}")

        if combined and len(captured_images) > 1:
            combined_screenshot = np.hstack(captured_images)
            cv2.imshow('Combined Screens', combined_screenshot)
        elif captured_images:
            for i, image in enumerate(captured_images):
                cv2.imshow(f'Monitor {screen_numbers[i] + 1}', image)

        if cv2.waitKey(1000 // frame_rate) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def find_image_center_in_region(captured_image, template_path, confidence=0.8):
    """
    Search for an image within a captured screen region and returns the center coordinates.

    Args:
        captured_image (numpy.ndarray): The captured screen region as a NumPy array.
        template_path (str): Path to the template image to search for.
        confidence (float): The confidence level for the match (between 0 and 1).

    Returns:
        Tuple[int, int] or None: The center coordinates of the found image, or None if not found.
    """
    # Load the template image
    template = cv2.imread(template_path, 0)
    if template is None:
        raise FileNotFoundError(f"Template image at {template_path} not found.")

    template_height, template_width = template.shape[:2]

    # Convert captured image to grayscale
    gray_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

    # Apply template matching
    res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Check if the maximum match value exceeds the confidence level
    if max_val >= confidence:
        # Calculate the center coordinates
        center_x = max_loc[0] + template_width // 2
        center_y = max_loc[1] + template_height // 2
        return center_x, center_y
    return None
