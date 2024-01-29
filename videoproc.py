import cv2
import os
import sys
import config
import imgproc

# Function that takes in a video, and outputs a sequence of images into a new directory
def extract_frames(video_path, output_folder, size_format="lowres"):
    """
    Function takes in a video file path, 
    outputs a sequence of images into a new directory
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    print(f"Using size format {size_format}")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read and save each frame
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if no more frames are available

        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        
        if size_format == "lowres":
            cv2.imwrite(frame_filename, imgproc.image_resize(frame, scale_factor=1/config.upscale_factor))
        else:
            cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    video_capture.release()

    print(f"Frames extracted successfully. Total frames: {frame_count}")




def frames_to_video(input_folder, output_video_path, fps=30):
    """
    Function that takes a directory of images
    Outputs a video file containing the sequence of frames
    """
    # Get the list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Check if there are any image files in the folder
    if not image_files:
        print("No image files found in the specified folder.")
        return

    # Get the first image to determine the frame size
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Initialize video writer
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    if not video_writer.isOpened():
        print("Error: Could not open video writer.")
        return

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            video_writer.write(frame)
        else:
            print("frame is None")

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully: {output_video_path}")