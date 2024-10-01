import cv2
import os

# Path to the folder containing the images
folder_path = '/Users/kinshukbansal/Desktop/blur'  # Update the folder path

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # Add other image formats if needed
        image_path = os.path.join(folder_path, filename)
        
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply GaussianBlur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
        
        # Thresholding the image to get the bright areas
        _, thresh_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert grayscale image to BGR (color) for drawing
        contoured_image_red = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Loop through each contour
        for contour in contours:
            # Fit a minimum enclosing circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))  # Convert center to integer
            radius = int(radius)       # Convert radius to integer
            
            # Draw the circle on the image (red color, thickness 2)
            cv2.circle(contoured_image_red, center, radius, (0, 0, 255), 2)
        
        # Display the result using OpenCV's imshow
        cv2.imshow(f'Circular Contours of {filename}', contoured_image_red)
        cv2.waitKey(0)  # Wait for a key press to close the window
        
        # Save the output image if needed
        output_path = os.path.join(folder_path, f'circular_contoured_{filename}')
        cv2.imwrite(output_path, contoured_image_red)

# Close all OpenCV windows
cv2.destroyAllWindows()
