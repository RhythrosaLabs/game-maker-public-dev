import streamlit as st
import requests
import json

def render_scene(scene_data):
    response = requests.post('http://localhost:5000/render', json=scene_data)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

st.title("Blender Integration in Streamlit")

x = st.slider("Cube X location", -10.0, 10.0, 0.0)
y = st.slider("Cube Y location", -10.0, 10.0, 0.0)
z = st.slider("Cube Z location", -10.0, 10.0, 0.0)

scene_data = {
    "cube_location": (x, y, z)
}

if st.button("Render Scene"):
    with st.spinner("Rendering..."):
        image = render_scene(scene_data)
    if image:
        st.image(image)
