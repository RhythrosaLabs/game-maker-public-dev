import streamlit as st
import bpy
import tempfile
import os

class BlenderComponent:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def render_scene(self, scene_file):
        # Load the Blender scene
        bpy.ops.wm.open_mainfile(filepath=scene_file)

        # Set up rendering parameters
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        output_file = os.path.join(self.temp_dir, "render_output.png")
        bpy.context.scene.render.filepath = output_file

        # Render the scene
        bpy.ops.render.render(write_still=True)

        return output_file

    def modify_object(self, object_name, location):
        # Select the object
        obj = bpy.data.objects[object_name]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Modify the object's location
        obj.location = location

    def run(self):
        st.title("Blender Integration in Streamlit")

        uploaded_file = st.file_uploader("Choose a Blender file", type="blend")
        if uploaded_file is not None:
            # Save the uploaded file
            scene_file = os.path.join(self.temp_dir, "scene.blend")
            with open(scene_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Render options
            if st.button("Render Scene"):
                output_file = self.render_scene(scene_file)
                st.image(output_file)

            # Object modification options
            object_name = st.text_input("Enter object name to modify")
            x = st.slider("X location", -10.0, 10.0, 0.0)
            y = st.slider("Y location", -10.0, 10.0, 0.0)
            z = st.slider("Z location", -10.0, 10.0, 0.0)

            if st.button("Modify Object"):
                self.modify_object(object_name, (x, y, z))
                st.success(f"Modified {object_name}'s location to ({x}, {y}, {z})")

if __name__ == "__main__":
    component = BlenderComponent()
    component.run()
