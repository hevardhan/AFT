import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):
    # Get the desktop window handle
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        # Capture only the left half of Screen 1
        full_width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        full_height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)

        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        # Capture half the width (left side)
        width = full_width // 4
        height = full_height* 2 // 3

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    # Cleanup
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Keep the image in color

def color_selection(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)
    
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    
    thresholds = (image[:,:,0] < rgb_threshold[0]) \
                | (image[:,:,1] < rgb_threshold[1]) \
                | (image[:,:,2] < rgb_threshold[2])
    
    color_select[thresholds] = [0,0,0]
    
    return color_select

def region_masking(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    region_select = np.copy(image)
    line_image = np.copy(image)

    # Define color selection criteria
    # MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Define the vertices of a triangular mask.
    # Keep in mind the origin (x=0, y=0) is in the upper left
    # MODIFY THESE VALUES TO ISOLATE THE REGION 
    # WHERE THE LANE LINES ARE IN THE IMAGE
    left_bottom = [100, 539]
    right_bottom = [950, 539]
    apex = [480, 290]

    # Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
    # np.polyfit returns the coefficients [A, B] of the fit
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Mask pixels below the threshold
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                        (image[:,:,1] < rgb_threshold[1]) | \
                        (image[:,:,2] < rgb_threshold[2])

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))
                        
    # Mask color and region selection
    # color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
    # Color pixels red where both color and region selections met
    line_image[~color_thresholds & region_thresholds] = [9, 255, 0]
    return line_image, image

def lane_detect(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = 180
    high_threshold = 240
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(280,450),(400,290), (440, 290), (582,450)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 2     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    lines_edges = cv2.polylines(lines_edges,vertices, True, (0,0,255), 10)

    return lines_edges

# Callback function to display coordinates dynamically
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Draw coordinates on the image dynamically
        display_img = param.copy()
        cv2.putText(display_img, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Left Half of Screen 1 (on Screen 2, Smaller)", display_img)

cv2.namedWindow("Left Half of Screen 1 (on Screen 2, Smaller)")

while True:
    left_half = grab_screen()

    # Resize the captured image
    resized_img = cv2.resize(left_half, (0, 0), fx=1, fy=1)

    # Apply lane detection
    resized_img = lane_detect(resized_img)

    # Add a title
    text = "RL Self Driving Truck"
    position = (10, resized_img.shape[0] - 10)
    cv2.putText(resized_img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Set mouse callback to show coordinates
    cv2.setMouseCallback("Left Half of Screen 1 (on Screen 2, Smaller)", show_coordinates, resized_img)

    # Display the processed image
    cv2.imshow("Left Half of Screen 1 (on Screen 2, Smaller)", resized_img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()