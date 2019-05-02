
# Import necessary library:
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def convertToLaneRetain(rgb_threshold, image, color_select):
    # Identify pixels below the threshold
    thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] > rgb_threshold[1]) | (
                image[:, :, 2] > rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]
    return color_select


# # Build a Card Finding Pipeline

# TODO: Build your pipeline that will draw lines on the test_images
# then save them to the test_images directory.

# Kernel
kernel = np.ones((5, 5), np.uint8)

# GAUSSIAN BLUR PARAMETERS
kernel_size = 23

# CANNY EDGE DETECTION PARAMETERS 70 - 140
low_threshold = 70
high_threshold = 140

# HOUGH LINES PARAMETERS
rho = 2
theta = np.pi / 180
threshold = 7
min_line_len = 40
max_line_gap = 20

# Define our color selection criteria
# Great number for both yellow and white lane
red_threshold = 255
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]


# TODO: Build your pipeline that will draw lines on the test_images
# then save them to the test_images directory.

# Kernel
kernel = np.ones((5, 5), np.uint8)

# GAUSSIAN BLUR PARAMETERS
kernel_size = 23

# HOUGH LINES PARAMETERS
rho = 2
theta = np.pi / 180
threshold = 7
min_line_len = 40
max_line_gap = 20

# Define our color selection criteria
# Great number for both yellow and white lane
red_threshold = 255
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]


def processed_img(img):
    gray = grayscale(img)
    t = cv2.mean(gray)[0]
    if t > 100:  # White background  - reduce the threshold
        # equalize the gray image
        gray = cv2.equalizeHist(gray)
        # CANNY EDGE DETECTION PARAMETERS 60 - 100
        low_threshold = 0.37 * cv2.mean(gray)[0]
        high_threshold = 1.8 * low_threshold
    else:
        #CANNY EDGE DETECTION PARAMETERS 70 - 140
        low_threshold = 0.6677 * cv2.mean(gray)[0]
        high_threshold = 1.8 * low_threshold
    print(low_threshold, high_threshold,t)

    # Blur the image
    img_blur = gaussian_blur(gray, kernel_size)

    # find edges
    edges = cv2.Canny(img_blur, low_threshold, high_threshold)

    # find lines on image
    line_image = np.copy(img) * 0
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_len, max_line_gap)

    # Iterate over the output "lines" and draw lines on the blank
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    # Convert color point to only solid black and white
    black = np.copy(combo)  # copy combo
    black = convertToLaneRetain(rgb_threshold, combo, black)  # Keep only red color
    split_black = np.copy(black[:, :, 0])

    # Connecting holes
    closing = cv2.morphologyEx(split_black, cv2.MORPH_CLOSE, kernel)

    return closing


# Parameter to dilated image
x = 60
y = 60
iterations = 2


def dilated_erosion(img):

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y))
    dilated = cv2.dilate(img, kernel_dilate, iterations=iterations)  # dilate, more the iteration more the dilation
    return dilated


def draw_shape(erosion, canvas):
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    contours, _ = cv2.findContours(erosion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    width_list = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 300 and h > 300:  # Only draw rectangle that has width and height > 300
            width_list.append(w * h)
            med_area = sum(width_list) / len(width_list)
            if w * h > 0.8 * med_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(canvas, [box], 0, (0, 0, 255), 20)
    return canvas


imgs = []
path = "train"
valid_images = [".jpg", ".gif", ".png", ".tga"]

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(os.path.join(path, f))

for index, i in enumerate(imgs):
    img = cv2.imread(i)
    ori = img.copy()
    blackWhite_image_new = processed_img(img)
    dilated_img_new = dilated_erosion(blackWhite_image_new)
    erosion = cv2.erode(dilated_img_new, kernel, iterations=5)
    shape = draw_shape(erosion, img)
    vis = np.concatenate((ori, shape), axis=1)
    cv2.imwrite('./output/' + 'output' + str(index) + '.png', vis)

    plt.figure(figsize=(10, 5))
    plt.imshow(vis, cmap='gray')
    # plt.title(i)
plt.show()

