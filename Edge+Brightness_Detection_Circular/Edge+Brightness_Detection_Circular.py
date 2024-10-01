import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Path to the folder containing the images
folder_path = '/Users/kinshukbansal/Desktop/abc'  # Update with the correct folder path

# Define the maximum radius for circles and bright region threshold
max_radius = 50  # Set the maximum radius for small circles
brightness_threshold = 200  # Threshold for the bright region

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # Add other image formats if needed
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
        
        # Convert grayscale image to BGR (color) for drawing
        combined_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
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
                    # Draw the circle on the image (red color, thickness 2)
                    cv2.circle(combined_image, center, radius, (0, 0, 255), 2)
        
        # Overlay the edges on the image in white for better visibility
        combined_image[edges > 0] = [255, 255, 255]
        
        # Save the output image
        output_path = os.path.join(folder_path, f'combined_{filename}')
        cv2.imwrite(output_path, combined_image)

        # Optionally display the result (comment out if not needed)
        plt.figure(figsize=(10, 6))
        plt.imshow(combined_image, cmap='gray')
        plt.title(f'Processed {filename}')
        plt.axis('off')
        plt.show()

# Close any OpenCV windows (if opened during debugging)
cv2.destroyAllWindows()
