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
                
                # Get the bounding box of the bright region
                x_bright, y_bright, w_bright, h_bright = cv2.boundingRect(contour)
                bright_region_dimensions = (w_bright, h_bright)
                
                # Crop the 4x4 matrix around the center of the detected circle
                # Ensure the coordinates for the crop are within image boundaries
                cropped_matrix = image[max(center[1] - 2, 0):center[1] + 2, max(center[0] - 2, 0):center[0] + 2]
                
                # Check if we have a valid 4x4 matrix, otherwise pad with zeros
                if cropped_matrix.shape != (4, 4):
                    padded_matrix = np.zeros((4, 4), dtype=np.uint8)
                    padded_matrix[:cropped_matrix.shape[0], :cropped_matrix.shape[1]] = cropped_matrix
                    cropped_matrix = padded_matrix
                
                # Print the 4x4 matrix in the grid-like format
                print(f"4x4 Intensity Matrix for {filename}:")
                for row in cropped_matrix:
                    print(row)
                
                # Print the dimensions of the bright region
                print(f"Dimensions of Bright Region (Width x Height): {bright_region_dimensions}")

                # Store the data
                all_circle_data.append({
                    'Image': filename,
                    'Center (x, y)': center,
                    'Radius': radius,
                    'Pixel Intensity': pixel_intensity,
                    'Distance from Origin': distance_from_origin,
                    '4x4 Intensity Matrix': cropped_matrix,
                    'Bright Region Dimensions': bright_region_dimensions
                })

# Convert the collected data into a pandas DataFrame
circle_df = pd.DataFrame(all_circle_data)

# Output the data
print(circle_df)

# Optionally, save the data to a CSV file
circle_df.to_csv('circle_data_output_with_4x4_matrix_and_bright_region_dimensions.csv', index=False)
