import streamlit as st
import requests
import time

API_URL = "http://your-flask-api-url.com"  # Replace with your actual API URL

def submit_render_job(scene_data):
    response = requests.post(f"{API_URL}/submit_job", json=scene_data)
    return response.json()['job_id']

def check_job_status(job_id):
    response = requests.get(f"{API_URL}/job_status/{job_id}")
    return response.json()['status']

def get_render_result(job_id):
    response = requests.get(f"{API_URL}/render_result/{job_id}", stream=True)
    if response.status_code == 200:
        return response.content
    return None

st.title("Blender Rendering Service")

x = st.slider("Cube X location", -10.0, 10.0, 0.0)
y = st.slider("Cube Y location", -10.0, 10.0, 0.0)
z = st.slider("Cube Z location", -10.0, 10.0, 0.0)

scene_data = {
    "cube_location": (x, y, z)
}

if st.button("Render Scene"):
    with st.spinner("Submitting render job..."):
        job_id = submit_render_job(scene_data)
    
    status = "pending"
    while status != "completed":
        time.sleep(5)  # Poll every 5 seconds
        status = check_job_status(job_id)
        st.text(f"Job status: {status}")
        if status == "failed":
            st.error("Rendering failed. Please try again.")
            break
    
    if status == "completed":
        image = get_render_result(job_id)
        if image:
            st.image(image)
        else:
            st.error("Failed to retrieve the rendered image.")
