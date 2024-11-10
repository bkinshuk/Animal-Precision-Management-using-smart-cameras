import cv2
import numpy as np
import math
import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the folder containing the images
folder_path = '/Users/kinshukbansal/Desktop/30_frames_for motion_detection'  # Folder path updated

# Define the maximum radius for circles and bright region threshold
max_radius = 50  # Set the maximum radius for small circles
brightness_threshold = 200  # Threshold for the bright region

# Data storage for all images and motion vectors
all_circle_data = []
motion_vectors = []

# Get all image filenames and sort them to maintain order
image_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")])

# Assuming all images have the same size, we will use the first image to get the size
first_image_path = os.path.join(folder_path, image_filenames[0])
first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
image_height, image_width = first_image.shape  # Get the dimensions of the image

# Loop through each sorted file
previous_center = None  # Initialize to store the previous center for motion vector calculation

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
                # Calculate pixel intensity at the center
                pixel_intensity = image[center[1], center[0]]
                
                # Calculate distance from origin
                distance_from_origin = math.sqrt((center[0] - origin[0]) ** 2 + (center[1] - origin[1]) ** 2)
                
                # Store the data for the current frame
                all_circle_data.append({
                    'Image': filename,
                    'Center (x, y)': center,
                    'Radius': radius,
                    'Pixel Intensity': pixel_intensity,
                    'Distance from Origin': distance_from_origin
                })
                
                # If we have a previous center, calculate the motion vector
                if previous_center is not None:
                    dx = center[0] - previous_center[0]  # Change in x-coordinate
                    dy = center[1] - previous_center[1]  # Change in y-coordinate
                    motion_vectors.append({
                        'From Image': prev_filename,
                        'To Image': filename,
                        'Start (x, y)': previous_center,
                        'End (x, y)': center,
                        'Motion Vector (dx, dy)': (dx, dy),
                        'Magnitude': math.sqrt(dx**2 + dy**2)
                    })
                
                # Update previous center and filename for the next iteration
                previous_center = center
                prev_filename = filename

# Convert the collected data into pandas DataFrames
circle_df = pd.DataFrame(all_circle_data)
motion_vector_df = pd.DataFrame(motion_vectors)

# Save the data to CSV files
csv_circle_data_path = 'circle_data_output.csv'
csv_motion_vectors_path = 'motion_vectors_output.csv'
circle_df.to_csv(csv_circle_data_path, index=False)
motion_vector_df.to_csv(csv_motion_vectors_path, index=False)

print(f"Circle data has been saved to {csv_circle_data_path}")
print(f"Motion vector data has been saved to {csv_motion_vectors_path}")

# Plot the coordinates of the circle centers with the same size as the image
plt.figure(figsize=(8, 6))
plt.scatter(circle_df['Center (x, y)'].apply(lambda x: x[0]), circle_df['Center (x, y)'].apply(lambda x: x[1]), color='blue', label='Circle Centers')

# Plot the motion vectors (arrows) between frames
for _, row in motion_vector_df.iterrows():
    start = row['Start (x, y)']
    end = row['End (x, y)']
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
              head_width=5, head_length=10, fc='red', ec='red', label='Motion Vectors')

plt.title('Circle Center Coordinates and Motion Vectors')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, image_width)  # Set the x-axis limit to match image width
plt.ylim(image_height, 0)  # Set the y-axis limit to match image height (inverted to match image coordinates)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio matches the image
plt.legend()

# Save the plot as an image file (PNG)
plot_output_path = 'circle_and_motion_vectors_plot.png'
plt.savefig(plot_output_path)
plt.show()

print(f"Plot with motion vectors has been saved to {plot_output_path}")
