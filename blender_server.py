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
