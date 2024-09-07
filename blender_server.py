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
