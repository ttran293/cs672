import cv2
import numpy as np
import matplotlib.pyplot as plt

### Question 3
### Task 2: Loading and Displaying Images

### Task 2.1: Read in the image
original_img = cv2.imread('elephant.jpeg', cv2.IMREAD_COLOR)
cv2.imshow('Elephant', original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Task 2.2: Display image using mathplotlib
plt.imshow(original_img)
plt.show()
cv2.imwrite('elephant_opencv.png', original_img)

### Task 2.3: Load elephant image with correct colors for matplotlib
rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.show()
cv2.imwrite('elephant_matplotlib.png', rgb_img)

### Task 3 Basic Image Processing Operations

### Task 3.1: Convert image to grayscale
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.show()
cv2.imwrite('elephant_gray.png', gray_img)

### Task 3.2: Cropping the baby elephant
cropped_img = original_img[350:945, 90:575] # y1:y2, x1:x2
matplotlib_img_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
plt.imshow(matplotlib_img_cropped)
plt.show()
cv2.imwrite('elephant_baby.png', cropped_img)

### Task 3.3: Resizing

### Task 3.3.a: Downsample image by 10 in width and height
downsampled_img = cv2.resize(original_img, (0, 0), fx=0.1, fy=0.1)
matplotlib_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
plt.imshow(matplotlib_img)
plt.show()
cv2.imwrite('elephant_10xdown.png', downsampled_img)

### Task 3.3.b: Upsample image 10x back to its original resolution using nearest neighbor and bicubic
elephant_10xup_nn = cv2.resize(downsampled_img, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
elephant_10xup_bicubic = cv2.resize(downsampled_img, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
matplotlib_img_10xup_nn = cv2.cvtColor(elephant_10xup_nn, cv2.COLOR_BGR2RGB)
matplotlib_img_10xup_bicubic = cv2.cvtColor(elephant_10xup_bicubic, cv2.COLOR_BGR2RGB)
plt.imshow(matplotlib_img_10xup_nn)
plt.show()
plt.imshow(matplotlib_img_10xup_bicubic)
plt.show()
cv2.imwrite('elephant_10xup_nn.png', elephant_10xup_nn)
cv2.imwrite('elephant_10xup_bicubic.png', elephant_10xup_bicubic)

### Task 3.3.c: Calculate the absolute difference between the ground truth and the two upsampled images
absolute_diff_nn = cv2.absdiff(original_img, elephant_10xup_nn)
absolute_diff_bicubic = cv2.absdiff(original_img, elephant_10xup_bicubic)
cv2.imwrite('elephant_diff_nn.png', absolute_diff_nn)
cv2.imwrite('elephant_diff_bicubic.png', absolute_diff_bicubic)

error_nn = np.sum(absolute_diff_nn.astype(np.int64))
error_bicubic = np.sum(absolute_diff_bicubic.astype(np.int64))

print(f"Error using nearest neighbor: {error_nn}")
print(f"Error using bicubic: {error_bicubic}")

### Question 4 Edge Detection and Image Blurring by Convolution
### Task 4.1: Write a function using Python and OpenCV to implement edge detection using 2D convolution
def edge_detection(image):
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

### Task 4.2: Repeat step 1, but with a different edge detection filter (for example, you can use a larger filter size, or different filter values)
def sobel_edge_detection(image):
    kernel = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)
    
### Task 4.3: Implement image blurring (also known as smoothing) using the same OpenCV function you used for edge detection. 
# Show the filter that you used for this operation, written as a matrix  
def image_blurring(image):
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.float32)/9
    return cv2.filter2D(image, -1, kernel)

### Task 4.4: Apply different blurring Gaussian filters
def gaussian_blurring(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

### Task 4.5: Display the original elephant image, the two edge detection outputs, and the two blurring
# outputs, side by side
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')
axs[0, 1].imshow(cv2.cvtColor(edge_detection(original_img), cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Edge Detection')
axs[0, 2].imshow(cv2.cvtColor(sobel_edge_detection(original_img), cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('Sobel Edge Detection')
axs[1, 0].imshow(cv2.cvtColor(image_blurring(original_img), cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Image Blurring')
axs[1, 1].imshow(cv2.cvtColor(gaussian_blurring(original_img), cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Gaussian Blurring')
axs[1, 2].axis('off')
plt.show()

### Task 4.6: Convolution from scratch
# Implement convolution without using any OpenCV or Numpy
# functions, but purely with Python. Apply it to the image and show the output with one edge
# detection filter and one blurring filter of your choice.
def convolution(image, kernel):
    k = len(kernel)
    flipped = [row[::-1] for row in kernel[::-1]]
    h, w = len(image), len(image[0])
    out_h, out_w = h - k + 1, w - k + 1
    output = []
    for i in range(out_h):
        row = []
        for j in range(out_w):
            total = 0.0
            for ki in range(k):
                for kj in range(k):
                    total += image[i + ki][j + kj] * flipped[ki][kj]
            if total < 0:
                total = -total
            total = max(0, min(255, total))
            row.append(int(total))
        output.append(row)
    return output


edge_kernel = [[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]]
blur_kernel = [[1/9, 1/9, 1/9],
               [1/9, 1/9, 1/9],
               [1/9, 1/9, 1/9]]
gray_list = gray_img.tolist()
conv_image_edge = convolution(gray_list, edge_kernel)
conv_image_blur = convolution(gray_list, blur_kernel)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(conv_image_edge, cmap='gray')
axs[0].set_title('Edge Detection Convolution')
axs[1].imshow(conv_image_blur, cmap='gray')
axs[1].set_title('Blurring Convolution')
plt.show()