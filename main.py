# blender_server.py
from flask import Flask, request, send_file
import subprocess
import os
import tempfile
import json

app = Flask(__name__)

BLENDER_PATH = "/path/to/blender"  # Update this with your Blender executable path

@app.route('/render', methods=['POST'])
def render_scene():
    scene_data = request.json
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_file = os.path.join(tmpdir, "scene.blend")
        output_file = os.path.join(tmpdir, "render_output.png")
        
        # Write scene data to a temporary file
        with open(scene_file, 'w') as f:
            json.dump(scene_data, f)
        
        # Call Blender to render the scene
        subprocess.run([
            BLENDER_PATH,
            "-b",  # background mode
            "-P", "render_script.py",  # Python script to execute
            "--",  # Separate Blender args from script args
            scene_file,
            output_file
        ])
        
        return send_file(output_file, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)

# render_script.py
import bpy
import sys
import json

def main():
    scene_file = sys.argv[-2]
    output_file = sys.argv[-1]
    
    with open(scene_file, 'r') as f:
        scene_data = json.load(f)
    
    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Create a simple scene based on the input data
    bpy.ops.mesh.primitive_cube_add(location=scene_data.get('cube_location', (0, 0, 0)))
    
    # Set up rendering parameters
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_file
    
    # Render the scene
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()

# streamlit_app.py
import streamlit as st
import requests
import json

def render_scene(scene_data):
    response = requests.post('http://localhost:5000/render', json=scene_data)
    return response.content

st.title("Blender Integration in Streamlit")

x = st.slider("Cube X location", -10.0, 10.0, 0.0)
y = st.slider("Cube Y location", -10.0, 10.0, 0.0)
z = st.slider("Cube Z location", -10.0, 10.0, 0.0)

scene_data = {
    "cube_location": (x, y, z)
}

if st.button("Render Scene"):
    image = render_scene(scene_data)
    st.image(image)
