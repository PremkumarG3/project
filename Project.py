#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary packages 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert it into RGB
image_path = 'taj mah.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the original image
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Create an ROI mask
roi_mask = np.zeros_like(img_rgb)
roi_mask[100:250, 100:250, :] = 255  # Define the ROI region

# Segment the ROI using bitwise AND operation
segmented_roi = cv2.bitwise_and(img_rgb, roi_mask)

# Display the segmented ROI
plt.imshow(segmented_roi)
plt.title('Segmented ROI')
plt.axis('off')
plt.show()


# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image at '{image_path}'")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
image_path = 'prem.jpg'
detect_handwriting(image_path)


# In[ ]:




