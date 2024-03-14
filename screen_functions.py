import ctypes
import numpy as np
import cv2
from display_structures import MONITORENUMPROC, BITMAPINFO, BITMAPINFOHEADER

# Load dlls
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
shcore = ctypes.windll.shcore
kernel32 = ctypes.windll.kernel32

SRCCOPY = 0xCC0020
PROCESS_PER_MONITOR_DPI_AWARE = 2

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
except AttributeError:
    # Fallback if SetProcessDpiAwareness does not exist (earlier versions of Windows)
    ctypes.windll.user32.SetProcessDPIAware()

def monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData, monitors):
    """
    Callback function for EnumDisplayMonitors.
    This function is called by the system for each monitor detected, adding the monitor's
    coordinates to the provided list.

    Args:
        hMonitor (HMONITOR): Handle to the display monitor.
        hdcMonitor (HDC): Handle to a device context.
        lprcMonitor (POINTER(RECT)): Pointer to a RECT structure with monitor coordinates.
        dwData (LPARAM): Application-defined data (unused in this function).
        monitors (list): List to append monitor coordinates.

    Returns:
        BOOL: True to continue enumeration, False to stop.
    """
    r = lprcMonitor.contents
    monitors.append((r.left, r.top, r.right, r.bottom))
    return True

def enumerate_monitors():
    """
    Enumerates all display monitors and returns their coordinates.

    Returns:
        list of tuples: A list where each tuple contains the coordinates (left, top, right, bottom)
        of a monitor.
    """
    monitors = []
    # Create a lambda function that adapts the callback signature and includes the monitors list.
    callback = MONITORENUMPROC(lambda hMonitor, hdcMonitor, lprcMonitor, dwData: 
                               monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData, monitors))
    # Load user32 DLL and call EnumDisplayMonitors with the callback.
    user32.EnumDisplayMonitors(None, None, callback, 0)
    return monitors

def capture_display(screen_numbers, region = None):
    """
    This function uses low level calls to access relevant display 
    information and returns a numpy array for the associated display
    or displays.
    Args:
        screen_numbers (List[int]): The list of screen numbers to capture.
        region (Optional[Tuple[Tuple[int, int], Tuple[int, int]]], optional): 
            A tuple of two tuples, each representing a point (x, y).
            For example, ((x1, y1), (x2, y2)) defines the top-left and bottom-right points of the region.
            Captures the full screen if None.

    Returns:
        numpy.ndarray or None: Captured image or None if failed.
    """
    monitor_list = enumerate_monitors()
    captured_images = []

    for screen_number in screen_numbers:
        if screen_number >= len(monitor_list):
            print(f"Invalid screen number: {screen_number}")
            continue

        monitor = monitor_list[screen_number]
        # Use the entire monitor size if no region is specified
        if region is None:
            left, top, right, bottom = monitor
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

        # Create a device context for the entire screen and a compatible context
        h_desktop_dc = user32.GetWindowDC(0)
        if not h_desktop_dc:
            print("GetWindowDC failed:", kernel32.GetLastError())
            return None
        
        h_capture_dc = gdi32.CreateCompatibleDC(h_desktop_dc)
        if not h_capture_dc:
            print("CreateCompatibleDC failed:", kernel32.GetLastError())
            user32.ReleaseDC(0, h_desktop_dc)
            return None
        
        h_capture_bitmap = gdi32.CreateCompatibleBitmap(h_desktop_dc, width, height)
        if not h_capture_bitmap:
            print("CreateCompatibleBitmap failed:", kernel32.GetLastError())
            gdi32.DeleteDC(h_capture_dc)
            user32.ReleaseDC(0, h_desktop_dc)
            return None
        
        gdi32.SelectObject(h_capture_dc, h_capture_bitmap)

        # Copy the screen content to the bitmap
        if not gdi32.BitBlt(h_capture_dc, 0, 0, width, height, h_desktop_dc, left, top, SRCCOPY):
            print("BitBlt failed:", kernel32.GetLastError())

        # Create a buffer and copy the bitmap data into it
        image_data = ctypes.create_string_buffer(width * height * 4)
        bmp_info = BITMAPINFO()
        bmp_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmp_info.bmiHeader.biWidth = width
        bmp_info.bmiHeader.biHeight = -height  # Negative to indicate a top-down DIB
        bmp_info.bmiHeader.biPlanes = 1
        bmp_info.bmiHeader.biBitCount = 32
        bmp_info.bmiHeader.biCompression = 0
        gdi32.GetDIBits(h_capture_dc, h_capture_bitmap, 0, height, image_data, ctypes.byref(bmp_info), 0)

        # Clean up
        gdi32.DeleteObject(h_capture_bitmap)
        gdi32.DeleteDC(h_capture_dc)
        user32.ReleaseDC(0, h_desktop_dc)

        # Convert to an image (NumPy array)
        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
        captured_images.append(image)

    # Combine images if more than one is captured
    if len(captured_images) > 1:
        return np.hstack(captured_images)

    return captured_images[0] if captured_images else None

def screenshots(screen_numbers=[0], combined = False, region = None):
    captured_images = []
    for screen_number in screen_numbers:
        screenshot = capture_display([screen_number], region)
        if screenshot is not None and screenshot.shape[0] > 0 and screenshot.shape[1] > 0:
            captured_images.append(screenshot)
        else:
            print(f"Invalid screenshot dimensions for Screen {screen_number + 1}")

    if combined and len(captured_images) > 1:
        combined_screenshot = np.hstack(captured_images)
        cv2.imshow('Combined Screens', combined_screenshot)
    else:
        for i, image in enumerate(captured_images):
            cv2.imshow(f'Screen {screen_numbers[i] + 1}', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def screenshot(screen_numbers = None, combined = False, region= None):
    if screen_numbers is None:
        screen_numbers = [0]
    
    captured_images = []
    for screen_number in screen_numbers:
        screenshot = capture_display([screen_number], region)
        if screenshot is not None and screenshot.shape[0] > 0 and screenshot.shape[1] > 0:
            captured_images.append(screenshot)
        else:
            print(f"Invalid screenshot dimensions for Screen {screen_number + 1}")

    if combined and len(captured_images) > 1:
        combined_screenshot = np.hstack(captured_images)
        return combined_screenshot
    elif len(captured_images) == 1:
        return captured_images[0]
    else:
        # Handling the case where no valid screenshots were captured
        return None

def screen_recorder(screen_numbers=[0], frame_rate=30, combined=False):

    if combined:
        cv2.namedWindow('Combined Screens', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Combined Screens', 1280, 480)  # Adjust size as needed
    else:
        for screen_number in screen_numbers:
            cv2.namedWindow(f'Screen {screen_number + 1}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Screen {screen_number + 1}', 640, 480)

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
                cv2.imshow(f'Screen {screen_numbers[i] + 1}', image)

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
