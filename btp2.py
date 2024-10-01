import cv2
import os
import numpy as np

# Load the video file
video_path = '/Users/kinshukbansal/Desktop/Untitled.mp4'  # Update to your actual video path
output_frames_dir = 'video_frames'
output_contour_dir = 'frames_with_contours'

# Create output directories if they don't exist
if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

if not os.path.exists(output_contour_dir):
    os.makedirs(output_contour_dir)

# Function to detect circles with more sensitivity and tighter constraints
def detect_circle(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 2)

    # Tighter HoughCircles parameters to focus on small circles
    circles = cv2.HoughCircles(blurred_frame, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30,
                               param1=40, param2=15, minRadius=5, maxRadius=20)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return None

def reduce_resolution(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Open the video for frame processing
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print("Error: Could not open video file")
    exit()

original_fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_step = 1  # Process every frame for now

frame_count = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print(f"Error reading frame {frame_count}")
        break

    # Save every frame to test if saving works
    frame_filename = f"{output_frames_dir}/frame_{frame_count:04d}.png"
    cv2.imwrite(frame_filename, frame)  # Save each frame
    print(f"Saved frame {frame_count} at {frame_filename}")

    # Detect circles in the reduced frame
    reduced_frame = reduce_resolution(frame)
    circles = detect_circle(reduced_frame)

    if circles is not None:
        # If circles are detected, draw them in red
        for (x, y, r) in circles:
            cv2.circle(reduced_frame, (x, y), r, (0, 0, 255), 3)  # Draw the circle in red

        # Save frame with detected circles
        contour_frame_filename = f"{output_contour_dir}/frame_{frame_count:04d}.png"
        cv2.imwrite(contour_frame_filename, reduced_frame)
        print(f"Saved frame with circles {frame_count} at {contour_frame_filename}")
    else:
        print(f"No circles detected in frame {frame_count}")

    frame_count += 1

# Release the video capture object
video_capture.release()

print(f"Processed {frame_count} frames.")
