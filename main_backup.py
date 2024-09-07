import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image
import replicate
import random
import subprocess

# Constants
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
API_KEY_FILE = "api_key.json"

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'openai': None, 'replicate': None}

if 'customization' not in st.session_state:
    st.session_state.customization = {
        'image_types': ['Character', 'Enemy', 'Background', 'Object'],
        'script_types': ['Player', 'Enemy', 'Game Object', 'Level Background'],
        'image_count': {'Character': 1, 'Enemy': 1, 'Background': 1, 'Object': 2},
        'script_count': {'Player': 1, 'Enemy': 1, 'Game Object': 3, 'Level Background': 1},
        'use_replicate': {'convert_to_3d': False, 'generate_music': False},
        'code_types': {'unity': False, 'unreal': False, 'blender': False},
        'blender_fbx': False,
        'generate_elements': {
            'game_concept': True,
            'world_concept': True,
            'character_concepts': True,
            'plot': True,
            'storyline': False,
            'dialogue': False,
            'game_mechanics': False,
            'level_design': False
        }
    }


# Load API keys from a file
def load_api_keys():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as file:
            data = json.load(file)
            return data.get('openai'), data.get('replicate')
    return None, None

# Save API keys to a file
def save_api_keys(openai_key, replicate_key):
    with open(API_KEY_FILE, 'w') as file:
        json.dump({"openai": openai_key, "replicate": replicate_key}, file)

# Get headers for OpenAI API
def get_openai_headers():
    return {
        "Authorization": f"Bearer {st.session_state.api_keys['openai']}",
        "Content-Type": "application/json"
    }

# Generate content using OpenAI API
def generate_content(prompt, role):
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(CHAT_API_URL, headers=get_openai_headers(), json=data)
        response.raise_for_status()
        response_data = response.json()
        if "choices" not in response_data:
            error_message = response_data.get("error", {}).get("message", "Unknown error")
            return f"Error: {error_message}"

        content_text = response_data["choices"][0]["message"]["content"]
        return content_text

    except requests.RequestException as e:
        return f"Error: Unable to communicate with the OpenAI API: {str(e)}"

# Generate images using OpenAI's DALL-E API
def generate_image(prompt, size):
    data = {
        "model": "dall-e-3",
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "url"
    }

    try:
        response = requests.post(DALLE_API_URL, headers=get_openai_headers(), json=data)
        response.raise_for_status()
        response_data = response.json()
        if "data" not in response_data:
            error_message = response_data.get("error", {}).get("message", "Unknown error")
            return f"Error: {error_message}"

        if not response_data["data"]:
            return "Error: No data returned from API."

        image_url = response_data["data"][0]["url"]
        return image_url

    except requests.RequestException as e:
        return f"Error: Unable to generate image: {str(e)}"

# Convert image to 3D model using Replicate API
def convert_image_to_3d(image_url):
    headers = {
        "Authorization": f"Token {st.session_state.api_keys['replicate']}",
        "Content-Type": "application/json"
    }
    data = {
        "input": {"image": image_url},
        "model": "adirik/wonder3d"
    }

    try:
        response = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get('output', {}).get('url')
    except requests.RequestException as e:
        return f"Error: Unable to convert image to 3D model: {str(e)}"

# Generate music using Replicate's MusicGen
def generate_music(prompt):
    replicate_client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
    
    try:
        input_data = {
            "prompt": prompt,
            "model_version": "stereo-large",
            "output_format": "mp3",
            "normalization_strategy": "peak"
        }
        
        output = replicate_client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input=input_data
        )
        
        return output
    
    except Exception as e:
        return f"Error: Unable to generate music: {str(e)}"

# Generate multiple images based on customization settings
def generate_images(customization, game_concept):
    images = {}
    
    image_prompts = {
        'Character': "Create a highly detailed, front-facing character concept art for a 2D game. The character should be in a neutral pose, with clearly defined features and high contrast. The design should be suitable for 3d rigging and for animation, with clear lines and distinct colors.",
        'Enemy': "Design a menacing, front-facing enemy character concept art for a 2D game. The enemy should have a threatening appearance with distinctive features, and be suitable for 3d rigging and animation. The design should be highly detailed with a clear silhouette, in a neutral pose.",
        'Background': "Create a wide, highly detailed background image for a level of the game. The scene should include a clear distinction between foreground, midground, and background elements. The style should be consistent with the theme, with room for character movement in the foreground.",
        'Object': "Create a detailed object image for a 2D game. The object should be a key item with a transparent background, easily recognizable, and fitting the theme. The design should be clear, with minimal unnecessary details, to ensure it integrates well into the game environment."
    }
    
    sizes = {
        'Character': '1024x1792',
        'Enemy': '1024x1792',
        'Background': '1792x1024',
        'Object': '1024x1024'
    }

    for img_type in customization['image_types']:
        for i in range(customization['image_count'].get(img_type, 1)):
            prompt = f"{image_prompts[img_type]} The design should fit the following game concept: {game_concept}. Variation {i + 1}"
            size = sizes[img_type]
            image_url = generate_image(prompt, size)
            if customization['use_replicate']['convert_to_3d'] and img_type != 'Background':
                image_url = convert_image_to_3d(image_url)
            images[f"{img_type.lower()}_image_{i + 1}"] = image_url

    return images

# Generate scripts based on customization settings and code types
def generate_scripts(customization, game_concept):
    script_descriptions = {
        'Player': f"Script for the player character with WASD controls and space bar to jump or shoot. The character should fit the following game concept: {game_concept}",
        'Enemy': f"Script for an enemy character with basic AI behavior. The enemy should fit the following game concept: {game_concept}",
        'Game Object': f"Script for a game object with basic functionality. The object should fit the following game concept: {game_concept}",
        'Level Background': f"Script for the level background. The background should fit the following game concept: {game_concept}"
    }
    
    scripts = {}
    for script_type in customization['script_types']:
        for i in range(customization['script_count'].get(script_type, 1)):
            desc = f"{script_descriptions[script_type]} - Instance {i + 1}"
            
            if customization['code_types']['unity']:
                unity_script = generate_content(f"Create a comprehensive Unity C# script for {desc}. Include detailed comments, error handling, and optimize for performance.", "Unity game development")
                scripts[f"unity_{script_type.lower()}_script_{i + 1}.cs"] = unity_script
            
            if customization['code_types']['unreal']:
                unreal_script = generate_content(f"Create a comprehensive Unreal Engine C++ script for {desc}. Include detailed comments, error handling, and optimize for performance.", "Unreal Engine game development")
                scripts[f"unreal_{script_type.lower()}_script_{i + 1}.cpp"] = unreal_script
            
            if customization['code_types']['blender']:
                blender_script = generate_content(f"Create a comprehensive Blender Python script for {desc}. The script should create a detailed 3D model suitable for game development, including proper mesh topology, materials, and possibly animations. Include detailed comments and error handling.", "Blender 3D modeling and animation")
                scripts[f"blender_{script_type.lower()}_script_{i + 1}.py"] = blender_script
    
    return scripts

def generate_blender_fbx(blender_script, output_path):
    # Write the Blender script to a temporary file
    temp_script_path = "temp_blender_script.py"
    with open(temp_script_path, "w") as f:
        f.write(blender_script)
    
    # Append FBX export commands to the script
    with open(temp_script_path, "a") as f:
        f.write(f"""
import bpy

# Ensure we're in object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Export as FBX
bpy.ops.export_scene.fbx(filepath="{output_path}", use_selection=True)
""")
    
    # Run Blender in background mode to execute the script
    blender_command = [
        "blender",
        "--background",
        "--python", temp_script_path
    ]
    
    try:
        subprocess.run(blender_command, check=True, capture_output=True, text=True)
        print(f"FBX file generated successfully at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating FBX file: {e}")
        print(f"Blender output: {e.output}")
    finally:
        # Clean up the temporary script file
        os.remove(temp_script_path)

# Generate additional game elements
def generate_additional_elements(game_concept, elements_to_generate):
    additional_elements = {}
    
    if elements_to_generate.get('storyline'):
        additional_elements['storyline'] = generate_content(f"Create a detailed storyline for the following game concept: {game_concept}", "game narrative design")
    
    if elements_to_generate.get('dialogue'):
        additional_elements['dialogue'] = generate_content(f"Write sample dialogue for key characters in the following game concept: {game_concept}", "game dialogue writing")
    
    if elements_to_generate.get('game_mechanics'):
        additional_elements['game_mechanics'] = generate_content(f"Describe detailed game mechanics for the following game concept: {game_concept}", "game design")
    
    if elements_to_generate.get('level_design'):
        additional_elements['level_design'] = generate_content(f"Create a detailed level design document for the following game concept: {game_concept}", "game level design")
    
    return additional_elements

# Generate a complete game plan
def generate_game_plan(user_prompt, customization):
    game_plan = {}
    
    # Status updates
    status = st.empty()
    progress_bar = st.progress(0)
    
    def update_status(message, progress):
        status.text(message)
        progress_bar.progress(progress)

    # Generate game concept
    if customization['generate_elements']['game_concept']:
        update_status("Generating game concept...", 0.1)
        game_plan['game_concept'] = generate_content(f"Invent a new 2D game concept with a detailed theme, setting, and unique features based on the following prompt: {user_prompt}. Ensure the game has WASD controls.", "game design")
    
    # Generate world concept
    if customization['generate_elements']['world_concept']:
        update_status("Creating world concept...", 0.2)
        game_plan['world_concept'] = generate_content(f"Create a detailed world concept for the 2D game: {game_plan['game_concept']}", "world building")
    
    # Generate character concepts
    if customization['generate_elements']['character_concepts']:
        update_status("Designing characters...", 0.3)
        game_plan['character_concepts'] = generate_content(f"Create detailed character concepts for the player and enemies in the 2D game: {game_plan['game_concept']}", "character design")
    
    # Generate plot
    if customization['generate_elements']['plot']:
        update_status("Crafting the plot...", 0.4)
        game_plan['plot'] = generate_content(f"Create a plot for the 2D game based on the world and characters of the game: {game_plan.get('world_concept', '')} and {game_plan.get('character_concepts', '')}.", "plot development")
    
    # Generate images
    update_status("Generating game images...", 0.5)
    game_plan['images'] = generate_images(customization, game_plan.get('game_concept', ''))
    
    # Generate scripts
    update_status("Writing game scripts...", 0.7)
    game_plan['scripts'] = generate_scripts(customization, game_plan.get('game_concept', ''))
    
    # Generate additional elements
    update_status("Creating additional game elements...", 0.8)
    game_plan['additional_elements'] = generate_additional_elements(game_plan.get('game_concept', ''), customization['generate_elements'])
    
    # Optional: Generate music
    if customization['use_replicate']['generate_music']:
        update_status("Composing background music...", 0.9)
        music_prompt = f"Create background music for the game: {game_plan.get('game_concept', '')}"
        game_plan['music'] = generate_music(music_prompt)

    update_status("Game plan generation complete!", 1.0)

    return game_plan

# Streamlit app layout
st.title("Automate Your Game Dev")

# Move game concept input to the top
st.header("Game Concept")
user_prompt = st.text_area("Describe your game concept", "Enter a detailed description of your game here...")

# Sidebar
st.sidebar.title("Settings")

# API Key Inputs (in the sidebar)
api_tab, about_tab = st.sidebar.tabs(["API Keys", "About"])

with api_tab:
    openai_key = st.text_input("OpenAI API Key", value=st.session_state.api_keys['openai'], type="password")
    replicate_key = st.text_input("Replicate API Key", value=st.session_state.api_keys['replicate'], type="password")
    if st.button("Save API Keys"):
        save_api_keys(openai_key, replicate_key)
        st.session_state.api_keys['openai'] = openai_key
        st.session_state.api_keys['replicate'] = replicate_key
        st.success("API Keys saved successfully!")

with about_tab:
    st.write("""
    # About Automate Your Game Dev

    This app helps game developers automate various aspects of their game development process using AI. 
    
    Key features:
    - Generate game concepts, world designs, and character ideas
    - Create game assets including images and scripts for Unity, Unreal, and Blender
    - Optional 3D model conversion, music generation, and Blender FBX export
    - Customizable content generation for various game elements
    
    Powered by OpenAI's GPT-4 and DALL-E 3, plus various Replicate AI models.
    
    Created by [Your Name/Company]. For support, contact: support@example.com
    """)

# Main content area
st.header("Customization")

# Image Customization
st.subheader("Image Customization")
for img_type in st.session_state.customization['image_types']:
    st.session_state.customization['image_count'][img_type] = st.number_input(
        f"Number of {img_type} Images", 
        min_value=1, 
        value=st.session_state.customization['image_count'][img_type]
    )

# Script Customization
st.subheader("Script Customization")
for script_type in st.session_state.customization['script_types']:
    st.session_state.customization['script_count'][script_type] = st.number_input(
        f"Number of {script_type} Scripts", 
        min_value=1, 
        value=st.session_state.customization['script_count'][script_type]
    )

# When setting checkbox values, use .get() method to avoid KeyError
st.session_state.customization['code_types']['unity'] = st.checkbox("Generate Unity C# Scripts", value=st.session_state.customization['code_types'].get('unity', False))
st.session_state.customization['code_types']['unreal'] = st.checkbox("Generate Unreal Engine C++ Scripts", value=st.session_state.customization['code_types'].get('unreal', False))
blender_col, fbx_col = st.columns([3, 2])
with blender_col:
    st.session_state.customization['code_types']['blender'] = st.checkbox("Generate Blender Python Scripts", value=st.session_state.customization['code_types'].get('blender', False))
with fbx_col:
    st.session_state.customization['blender_fbx'] = st.checkbox("Export FBX", disabled=not st.session_state.customization['code_types']['blender'], value=st.session_state.customization.get('blender_fbx', False))

# Replicate Options
st.subheader("Replicate Options")
st.session_state.customization['use_replicate']['convert_to_3d'] = st.checkbox("Convert Images to 3D")
st.session_state.customization['use_replicate']['generate_music'] = st.checkbox("Generate Music")

# Additional Elements Selection
st.subheader("Additional Game Elements")
st.session_state.customization['generate_elements']['storyline'] = st.checkbox("Generate Detailed Storyline")
st.session_state.customization['generate_elements']['dialogue'] = st.checkbox("Generate Sample Dialogue")
st.session_state.customization['generate_elements']['game_mechanics'] = st.checkbox("Generate Game Mechanics Description")
st.session_state.customization['generate_elements']['level_design'] = st.checkbox("Generate Level Design Document")

# Generate Game Plan
if st.button("Generate Game Plan"):
    if not st.session_state.api_keys['openai'] or not st.session_state.api_keys['replicate']:
        st.error("Please enter and save both OpenAI and Replicate API keys.")
    else:
        game_plan = generate_game_plan(user_prompt, st.session_state.customization)

        # Display game plan results
        if 'game_concept' in game_plan:
            st.subheader("Game Concept")
            st.write(game_plan['game_concept'])

        if 'world_concept' in game_plan:
            st.subheader("World Concept")
            st.write(game_plan['world_concept'])

        if 'character_concepts' in game_plan:
            st.subheader("Character Concepts")
            st.write(game_plan['character_concepts'])

        if 'plot' in game_plan:
            st.subheader("Plot")
            st.write(game_plan['plot'])

        st.subheader("Assets")
        st.write("### Images")
        for img_name, img_url in game_plan['images'].items():
            st.write(f"{img_name}: [View Image]({img_url})")
            if st.session_state.customization['use_replicate']['convert_to_3d'] and 'background' not in img_name.lower():
                st.write(f"3D Model: [View 3D Model]({convert_image_to_3d(img_url)})")
        
        st.write("### Scripts")
        for script_name, script_code in game_plan['scripts'].items():
            st.write(f"{script_name}:\n```\n{script_code}\n```")

        if 'additional_elements' in game_plan:
            st.subheader("Additional Game Elements")
            for element_name, element_content in game_plan['additional_elements'].items():
                st.write(f"### {element_name.capitalize()}")
                st.write(element_content)

        # Save results
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Add text documents
            for key in ['game_concept', 'world_concept', 'character_concepts', 'plot']:
                if key in game_plan:
                    zip_file.writestr(f"{key}.txt", game_plan[key])
            
            # Add images
            for img_name, img_url in game_plan['images'].items():
                if img_url.startswith('http'):
                    img_response = requests.get(img_url)
                    img = Image.open(BytesIO(img_response.content))
                    img_file_name = f"{img_name}.png"
                    with BytesIO() as img_buffer:
                        img.save(img_buffer, format='PNG')
                        zip_file.writestr(img_file_name, img_buffer.getvalue())
            
            # Add scripts
            for script_name, script_code in game_plan['scripts'].items():
                zip_file.writestr(script_name, script_code)
                
                # Generate FBX for Blender scripts if option is selected
                if st.session_state.customization['blender_fbx'] and script_name.startswith('blender_'):
                    fbx_name = script_name.replace('.py', '.fbx')
                    fbx_path = os.path.join(os.getcwd(), fbx_name)
                    generate_blender_fbx(script_code, fbx_path)
                    if os.path.exists(fbx_path):
                        with open(fbx_path, 'rb') as fbx_file:
                            zip_file.writestr(fbx_name, fbx_file.read())
                        os.remove(fbx_path)
            
            # Add additional elements
            if 'additional_elements' in game_plan:
                for element_name, element_content in game_plan['additional_elements'].items():
                    zip_file.writestr(f"{element_name}.txt", element_content)
            
            # Add music if generated
            if st.session_state.customization['use_replicate']['generate_music'] and 'music' in game_plan:
                music_response = requests.get(game_plan['music'])
                zip_file.writestr("background_music.mp3", music_response.content)

        st.download_button("Download ZIP of Assets and Scripts", zip_buffer.getvalue(), file_name="game_plan.zip")

        # Display generated music if applicable
        if st.session_state.customization['use_replicate']['generate_music'] and 'music' in game_plan:
            st.subheader("Generated Music")
            st.audio(game_plan['music'], format='audio/mp3')

# End of the Streamlit app
