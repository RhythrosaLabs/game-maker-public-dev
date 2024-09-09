import streamlit as st
import requests
import base64
from PIL import Image
import io
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import sys

# Redirect stderr to stdout to avoid issues with logging in some environments
sys.stderr = sys.stdout

# Function to resize image to supported dimensions
def resize_image(image):
    width, height = image.size
    if (width, height) in [(1024, 576), (576, 1024), (768, 768)]:
        return image  # Return if image already has valid dimensions
    else:
        st.warning("Resizing image to 768x768 (default)")
        return image.resize((768, 768))  # Default resize

# Function to generate image from text prompt
def generate_image_from_text(api_key, prompt):
    url = "https://api.stability.ai/v1beta/generation/stable-diffusion-v1-6/text-to-image"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 768,
        "width": 768,
        "samples": 1,
        "steps": 30,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        image_data = response.json()['artifacts'][0]['base64']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))  # Decode base64 to image
        return image
    else:
        st.error(f"Error generating image: {response.status_code} - {response.text}")
        return None

# Function to start video generation
def start_video_generation(api_key, image, cfg_scale=1.8, motion_bucket_id=127, seed=0):
    url = "https://api.stability.ai/v2beta/image-to-video"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Convert PIL image to bytes for the request
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {
        "image": ("image.png", img_byte_arr, "image/png")
    }

    data = {
        "seed": str(seed),
        "cfg_scale": str(cfg_scale),
        "motion_bucket_id": str(motion_bucket_id)
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code == 200:
        return response.json().get('id')
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Function to poll for video generation result
def poll_for_video(api_key, generation_id):
    url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "video/*"
    }

    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 202:
            st.write("Video generation in progress... Polling again in 10 seconds.")
        elif response.status_code == 200:
            return response.content
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None

# Function to validate if a video segment is valid (i.e., not corrupted)
def validate_video_clip(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration > 0
    except Exception as e:
        st.error(f"Invalid video segment: {video_path}, Error: {str(e)}")
        return False

# Function to ensure concatenation of valid videos only
def concatenate_videos(video_clips):
    valid_clips = []
    for clip_path in video_clips:
        if validate_video_clip(clip_path):
            valid_clips.append(VideoFileClip(clip_path))

    # Ensure we have valid clips to concatenate
    if not valid_clips:
        st.error("No valid video segments found. Unable to concatenate.")
        return None

    try:
        final_video = concatenate_videoclips(valid_clips)
        for clip in valid_clips:
            clip.close()  # Ensure each video file is properly closed
        return final_video
    except Exception as e:
        st.error(f"Error concatenating videos: {str(e)}")
        return None

# Function to extract the last frame from a video and convert it to an image
def get_last_frame_image(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        last_frame = video_clip.get_frame(video_clip.duration)
        last_frame_image = Image.fromarray(np.uint8(last_frame)).convert('RGB')
        video_clip.close()  # Close the video clip after extracting the frame
        return last_frame_image
    except Exception as e:
        st.error(f"Error extracting last frame from {video_path}: {str(e)}")
        return None

# Streamlit UI
def main():
    st.title("Stable Diffusion Longform Video Creator")

    # User inputs API key and options for text-to-video or image-to-video
    api_key = st.text_input("Enter your Stability AI API Key", type="password")
    mode = st.radio("Select Mode", ("Text-to-Video", "Image-to-Video"))

    if mode == "Text-to-Video":
        prompt = st.text_input("Enter a text prompt for video generation")
    else:
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # Parameters for video generation
    cfg_scale = st.slider("CFG Scale (Stick to original image)", 0.0, 10.0, 1.8)
    motion_bucket_id = st.slider("Motion Bucket ID (Less motion to more motion)", 1, 255, 127)
    seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0)
    num_segments = st.slider("Number of video segments to generate", 1, 60, 5)

    if st.button("Generate Longform Video"):
        if not api_key:
            st.error("Please enter the API key.")
            return

        if mode == "Text-to-Video" and not prompt:
            st.error("Please enter a text prompt.")
            return

        if mode == "Image-to-Video" and not image_file:
            st.error("Please upload an image.")
            return

        # Generate image from text or use uploaded image
        if mode == "Text-to-Video":
            st.write("Generating image from text prompt...")
            image = generate_image_from_text(api_key, prompt)
            if image is None:
                return
        else:
            image = Image.open(image_file)

        # Resize image if necessary
        image = resize_image(image)

        # Generate and concatenate videos
        video_clips = []
        current_image = image

        for i in range(num_segments):
            st.write(f"Generating video segment {i+1}/{num_segments}...")
            generation_id = start_video_generation(api_key, current_image, cfg_scale, motion_bucket_id, seed)

            if generation_id:
                video_content = poll_for_video(api_key, generation_id)

                if video_content:
                    # Save the video segment
                    video_path = f"video_segment_{i+1}.mp4"
                    with open(video_path, "wb") as f:
                        f.write(video_content)
                    video_clips.append(video_path)

                    # Extract the last frame from the video and convert it to an image
                    last_frame_image = get_last_frame_image(video_path)
                    if last_frame_image:
                        current_image = last_frame_image  # Update the image for the next segment
                else:
                    st.error("Failed to retrieve video content.")
                    return
            else:
                st.error("Failed to start video generation.")
                return

        if video_clips:
            # Concatenate all valid video segments
            st.write("Concatenating video segments into one longform video...")
            final_video = concatenate_videos(video_clips)
            if final_video:
                final_video_path = "longform_video.mp4"
                final_video.write_videofile(final_video_path, logger=None)  # Suppress logging
                
                # Provide download link
                st.video(final_video_path)
                with open(final_video_path, "rb") as f:
                    st.download_button("Download Longform Video", f, file_name="longform_video.mp4")
            
            # Clean up video files after concatenation
            for video_file in video_clips:
                os.remove(video_file)

if __name__ == "__main__":
    main()
