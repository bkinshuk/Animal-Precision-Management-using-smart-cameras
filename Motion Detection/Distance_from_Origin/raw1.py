import cv2
import numpy as np
import math
import os
import pandas as pd

# Path to the folder containing the images
folder_path = '/Users/kinshukbansal/Desktop/30_frames_for motion_detection'  # Update this path accordingly

# Define the maximum radius for circles and bright region threshold
max_radius = 50  # Set the maximum radius for small circles
brightness_threshold = 200  # Threshold for the bright region

# Data storage for all images
all_circle_data = []

# Get all image filenames and sort them to maintain order
image_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")])

# Loop through each sorted file
for filename in image_filenames:
    image_path = os.path.join(folder_path, filename)
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred_image, 100, 200)
    
    # Apply thresholding to find bright regions
    _, bright_regions = cv2.threshold(blurred_image, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours based on the edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Origin point for distance calculation
    origin = (0, 0)
    
    # Loop through each contour
    for contour in contours:
        # Fit a minimum enclosing circle around the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))  # Convert center to integer
        radius = int(radius)       # Convert radius to integer
        
        # Check if the region inside the contour is bright and the circle is small
        if radius <= max_radius:
            # Create a mask for the contour
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour on the mask
            
            # Check if the contour is within a bright region
            if np.any(cv2.bitwise_and(bright_regions, bright_regions, mask=mask)):
                # Ensure the pixel intensity is non-negative and valid
                pixel_intensity = image[center[1], center[0]]
                pixel_intensity = np.clip(pixel_intensity, 0, 255)  # Clamp values between 0 and 255
                
                # Calculate distance from origin
                distance_from_origin = math.sqrt((center[0] - origin[0]) ** 2 + (center[1] - origin[1]) ** 2)
                
                # Store the data
                all_circle_data.append({
                    'Image': filename,
                    'Center (x, y)': center,
                    'Radius': radius,
                    'Pixel Intensity': pixel_intensity,
                    'Distance from Origin': distance_from_origin
                })

# Convert the collected data into a pandas DataFrame
circle_df = pd.DataFrame(all_circle_data)

# Output the data
print(circle_df)

# Optionally, save the data to a CSV file
circle_df.to_csv('circle_data_output.csv', index=False)
