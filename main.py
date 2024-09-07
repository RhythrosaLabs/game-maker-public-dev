import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image
import replicate

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
        'image_types': ['Character', 'Enemy', 'Background', 'Object', 'Texture', 'Sprite', 'UI'],
        'script_types': ['Player', 'Enemy', 'Game Object', 'Level Background'],
        'image_count': {t: 0 for t in ['Character', 'Enemy', 'Background', 'Object', 'Texture', 'Sprite', 'UI']},
        'script_count': {t: 0 for t in ['Player', 'Enemy', 'Game Object', 'Level Background']},
        'use_replicate': {'convert_to_3d': False, 'generate_music': False},
        'code_types': {'unity': False, 'unreal': False, 'blender': False},
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
        "model": "gpt-4",
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
    replicate_client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
    
    try:
        output = replicate_client.run(
            "adirik/wonder3d:f7ecac6510cc161e3f2c9457cd164d57d49043d4dcbd261d9b21cd4a07ab9dfe",
            input={"image": image_url}
        )
        return output.get('3d_model')
    except Exception as e:
        return f"Error: Unable to convert image to 3D model: {str(e)}"

# Generate music using Replicate's MusicGen
def generate_music(prompt):
    replicate_client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
    
    try:
        output = replicate_client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": prompt,
                "model_version": "stereo-large",
                "output_format": "mp3",
                "normalization_strategy": "peak"
            }
        )
        return output
    except Exception as e:
        return f"Error: Unable to generate music: {str(e)}"

# Generate multiple images based on customization settings
def generate_images(customization, game_concept):
    images = {}
    
    image_prompts = {
        'Character': "Create a highly detailed, front-facing character concept art for a 2D game. The character should be in a neutral pose, with clearly defined features and high contrast. Design for easy game integration with no background, suitable for 3D conversion. The design should have clear lines and distinct colors.",
        'Enemy': "Design a menacing, front-facing enemy character concept art for a 2D game. The enemy should have a threatening appearance with distinctive features. Create with no background for easy game integration and 3D conversion. The design should be highly detailed with a clear silhouette, in a neutral pose.",
        'Background': "Create a wide, highly detailed background image for a level of the game. The scene should include a clear distinction between foreground, midground, and background elements. The style should be consistent with the theme, with room for character movement in the foreground.",
        'Object': "Create a detailed object image for a 2D game. The object should be a key item with a transparent background, easily recognizable, and fitting the theme. Design for easy game integration and potential 3D conversion. The design should be clear, with minimal unnecessary details.",
        'Texture': "Generate a seamless texture pattern suitable for use in a 2D game. The texture should be tileable and have a resolution of 512x512 pixels. Ensure the pattern is consistent and doesn't have visible seams when tiled.",
        'Sprite': "Create a game sprite sheet with multiple animation frames for a character or object. The sprite should be designed with a transparent background, clear silhouette, and distinct frames for easy animation in a 2D game engine.",
        'UI': "Design a user interface element for a 2D game. This could be a button, icon, health bar, or menu background. Ensure the design is clean, intuitive, and fits the game's overall aesthetic. Create with a transparent background for easy integration."
    }
    
    sizes = {
        'Character': '1024x1024',
        'Enemy': '1024x1024',
        'Background': '1792x1024',
        'Object': '1024x1024',
        'Texture': '1024x1024',
        'Sprite': '1024x1024',
        'UI': '1024x1024'
    }

    for img_type in customization['image_types']:
        for i in range(customization['image_count'].get(img_type, 0)):
            prompt = f"{image_prompts[img_type]} The design should fit the following game concept: {game_concept}. Variation {i + 1}"
            size = sizes[img_type]
            image_url = generate_image(prompt, size)
            if customization['use_replicate']['convert_to_3d'] and img_type in ['Character', 'Enemy', 'Object']:
                model_url = convert_image_to_3d(image_url)
                images[f"{img_type.lower()}_3d_model_{i + 1}"] = model_url
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
        for i in range(customization['script_count'].get(script_type, 0)):
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
    if any(customization['image_count'].values()):
        update_status("Generating game images...", 0.5)
        game_plan['images'] = generate_images(customization, game_plan.get('game_concept', ''))
    
    # Generate scripts
    if any(customization['script_count'].values()):
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

# Streamlit app layout (continued)

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
    - Optional 3D model conversion and music generation
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
        min_value=0, 
        value=st.session_state.customization['image_count'][img_type]
    )

# Script Customization
st.subheader("Script Customization")
for script_type in st.session_state.customization['script_types']:
    st.session_state.customization['script_count'][script_type] = st.number_input(
        f"Number of {script_type} Scripts", 
        min_value=0, 
        value=st.session_state.customization['script_count'][script_type]
    )

# Code Type Selection
st.subheader("Code Type Selection")
st.session_state.customization['code_types']['unity'] = st.checkbox("Generate Unity C# Scripts")
st.session_state.customization['code_types']['unreal'] = st.checkbox("Generate Unreal Engine C++ Scripts")
st.session_state.customization['code_types']['blender'] = st.checkbox("Generate Blender Python Scripts")

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

        if 'images' in game_plan:
            st.subheader("Assets")
            st.write("### Images")
            for img_name, img_url in game_plan['images'].items():
                if '3d_model' in img_name:
                    st.write(f"{img_name}: [View 3D Model]({img_url})")
                else:
                    st.write(f"{img_name}: [View Image]({img_url})")

        if 'scripts' in game_plan:
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
            
            # Add images and 3D models
            if 'images' in game_plan:
                for asset_name, asset_url in game_plan['images'].items():
                    if asset_url.startswith('http'):
                        asset_response = requests.get(asset_url)
                        if '3d_model' in asset_name:
                            zip_file.writestr(f"{asset_name}.glb", asset_response.content)
                        else:
                            img = Image.open(BytesIO(asset_response.content))
                            img_file_name = f"{asset_name}.png"
                            with BytesIO() as img_buffer:
                                img.save(img_buffer, format='PNG')
                                zip_file.writestr(img_file_name, img_buffer.getvalue())
            
            # Add scripts
            if 'scripts' in game_plan:
                for script_name, script_code in game_plan['scripts'].items():
                    zip_file.writestr(script_name, script_code)
            
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
