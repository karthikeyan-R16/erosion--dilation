# Implementation-of-Erosion-and-Dilation
## Aim
To implement Erosion and Dilation using Python and OpenCV.
## Software Required
1. Anaconda - Python 3.7
2. OpenCV
## Algorithm:
### Step1:
Import required libraries (OpenCV, NumPy) and load the image in grayscale.

### Step2:
Define a structuring element (kernel) for morphological operations.

### Step3:
Apply erosion using cv2.erode() on the image with the defined kernel.

### Step4:
Apply dilation using cv2.dilate() on the image with the same kernel.

### Step5:
Display and compare the original, eroded, and dilated images.
 
## Program:

``` Python
# Import the necessary packages

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a blank image
image = np.zeros((500, 500, 3), dtype=np.uint8)


# Create the Text using cv2.putText

# Add text on the image using cv2.putText
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'My Text', (100, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


# Create the structuring element

# Display the input image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
plt.title("Input Image with Text")
plt.axis('off')
plt.show()

# Create a simple square kernel (3x3)
kernel = np.ones((3, 3), np.uint8)

# Erode the image

eroded_image = cv2.erode(image, kernel, iterations=1)

# Display the eroded image
plt.imshow(cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.title("Eroded Image")
plt.axis('off')
plt.show()


# Dilate the image

# Apply dilation (expanding effect)
dilated_image = cv2.dilate(image, kernel, iterations=1)

# Display the dilated image
plt.imshow(cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.title("Dilated Image")
plt.axis('off')
plt.show()

```
## Output:

### Display the input Image
![image](https://github.com/user-attachments/assets/4bd5d70c-e6ab-4751-8045-1e69e12ce71b)

### Display the Eroded Image
![image](https://github.com/user-attachments/assets/e1d85525-68ce-4cb0-897e-b78a2ff753b5)

### Display the Dilated Image
![image](https://github.com/user-attachments/assets/ae53f289-227f-41f9-ae3e-4e1c361459f4)


## Result
Thus the generated text image is eroded and dilated using python and OpenCV.
