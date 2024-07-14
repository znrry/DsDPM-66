import os
import cv2

def video_to_frames(video_path, output_folder, frame_rate):
    # Create an output folder corresponding to the video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the video frame rate and total number of frames
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate how many frames should be spaced to capture an image
    interval = int(round(fps / frame_rate))

    # Reads video frame by frame and saves the image
    frame_count = 0
    image_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1

        # Keep only the frames with the specified frame rate
        if frame_count % interval == 0:
            output_path = os.path.join(output_folder, f"{video_name}_{image_count}.jpg")
            cv2.imwrite(output_path, frame)
            image_count += 1

    # Releasing the Video Capture Object
    video_capture.release()

    print(f"Video {video_path} cutting completion，total generated {image_count} images。")

# Setting the input folder and output folder paths
input_folder = "raw_video"
output_folder = "raw_images"

# Setting the frame rate
frame_rate = 2  # 10 frames per second

# Iterate over the video files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_folder, filename)
        video_to_frames(video_path, output_folder, frame_rate)