import cv2
import numpy as np

def addHorizontalLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # convert image to grayscale
    img_height, img_width = gray.shape  # get dimension of image

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    invert = False
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9*img_height and (w > max(10,img_width / 30) and h > max(10,img_height / 30))):
            invert = True
            img_bin[y:y+h,x:x+w] = 255 - img_bin[y:y+h,x:x+w]

    img_bin = 255 - img_bin if (invert) else img_bin

    img_bin_inv = 255 - img_bin
    ############################################################################################################################################  
    kernel_len_ver = max(10, img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver)) #shape (kernel_len, 1) inverted! xD

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1)) #shape (1,kernel_ken) xD
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)   
    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Eroding and thesholding the image
    img_vh = cv2.dilate(img_vh, kernel, iterations=3)
   
    thresh, img_vh = (cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY))

    bitor = cv2.bitwise_or(img_bin, img_vh)

    img_median = bitor #cv2.medianBlur(bitor, 3)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 3)) #shape (kernel_len, 1) inverted! xD
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    # get contours of horizontal lines
    contours, hierarchy = cv2.findContours(horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    hor = img.copy()  # image to draw horizontal line on
    for cntr in contours:
        x1,y1,x2,y2 = cv2.boundingRect(cntr)
        ycenter = y1+y2//2            
        cv2.line(hor, (0,ycenter), (img_width-1,ycenter), (0, 0, 0), 2)   # draw only horizontal lines on original image

    return hor
    