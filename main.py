import streamlit as st
import requests
import json
import os
import zipfile
import shutil
import tempfile
from io import BytesIO
from PIL import Image
import replicate
import base64
import re

# Constants
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
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
        'use_replicate': {'generate_music': False},
        'convert_to_3d': {t: False for t in ['Character', 'Enemy', 'Object', 'UI']},
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
        },
        'image_model': 'dall-e-3',
        'chat_model': 'gpt-4o-mini',
        'code_model': 'gpt-4o-mini',
        'procedural_generation': {
            'world': False,
            'levels': False,
            'items': False
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

# Generate content using selected chat model
def generate_content(prompt, role):
    if st.session_state.customization['chat_model'] in ['gpt-4', 'gpt-4o-mini']:
        data = {
            "model": st.session_state.customization['chat_model'],
            "messages": [
                {"role": "system", "content": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses optimized for game development."},
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
    elif st.session_state.customization['chat_model'] == 'llama':
        replicate_client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
        try:
            output = replicate_client.run(
                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                input={
                    "prompt": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses optimized for game development. User query: {prompt}",
                    "max_length": 500,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            return ''.join(output)
        except Exception as e:
            return f"Error: Unable to generate content using Llama: {str(e)}"
    else:
        return "Error: Invalid chat model selected."

# Generate images using selected image model
def generate_image(prompt, size):
    if st.session_state.customization['image_model'] == 'dall-e-3':
        data = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": f"{size[0]}x{size[1]}",
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

            return response_data["data"][0]["url"]

        except requests.RequestException as e:
            return f"Error: Unable to generate image: {str(e)}"
    elif st.session_state.customization['image_model'] == 'SD Flux-1':
        try:
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
            output = client.run(
                "black-forest-labs/flux-pro",
                input={
                    "prompt": prompt,
                    "aspect_ratio": "1:1" if size[0] == size[1] else "16:9",
                    "steps": 25,
                    "guidance": 3.0,
                    "interval": 2.0,
                    "safety_tolerance": 2,
                    "output_format": "png",
                    "output_quality": 100
                }
            )
            return output
        except Exception as e:
            return f"Error: Unable to generate image using SD Flux-1: {str(e)}"
    elif st.session_state.customization['image_model'] == 'SDXL Lightning':
        try:
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
            output = client.run(
                "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
                input={"prompt": prompt}
            )
            return output[0] if output else None
        except Exception as e:
            return f"Error: Unable to generate image using SDXL Lightning: {str(e)}"
    else:
        return "Error: Invalid image model selected."

# Convert image to 3D model using Replicate API
def convert_image_to_3d(image_url):
    try:
        client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
        output = client.run(
            "camenduru/lgm:d2870893aa115773465a823fe70fd446673604189843f39a99642dd9171e05e2",
            input={
                "input_image": image_url,
                "prompt": "a 3D model",
                "negative_prompt": "ugly, blurry, pixelated, obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate",
                "seed": 42
            }
        )
        
        result = {'glb': None, 'obj': None}
        for url in output:
            if url.endswith('.glb'):
                result['glb'] = url
            elif url.endswith('.obj'):
                result['obj'] = url
        
        return result if (result['glb'] or result['obj']) else None
    except Exception as e:
        st.error(f"Error during 3D conversion: {str(e)}")
        return None

# Generate music using Replicate's MusicGen
def generate_music(prompt):
    try:
        client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
        output = client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": prompt,
                "model_version": "stereo-large",
                "output_format": "mp3",
                "normalization_strategy": "peak"
            }
        )
        if isinstance(output, str) and output.startswith("http"):
            return output
        else:
            return None
    except Exception as e:
        st.error(f"Error: Unable to generate music: {str(e)}")
        return None

# Generate multiple images based on customization settings
def generate_images(customization, game_concept):
    images = {}
    
    image_prompts = {
        'Character': "Create a highly detailed, front-facing character concept art for a 2D game. The character should be in a T-pose with arms slightly away from the body, featuring clearly defined features and high contrast. Design for easy game integration with a transparent background, suitable for 3D conversion. The design should have clear outlines, distinct colors, and be easily animatable. Ensure all body parts are visible and not obscured.",
        'Enemy': "Design a menacing, front-facing enemy character concept art for a 2D game. The enemy should be in a T-pose with arms slightly away from the body, having a threatening appearance with distinctive features. Create with a transparent background for easy game integration and 3D conversion. The design should be highly detailed with a clear silhouette, easily recognizable, and suitable for animation.",
        'Background': "Create a wide, highly detailed background image for a level of the game. The scene should include a clear distinction between foreground, midground, and background elements, with parallax-ready layers. The style should be consistent with the theme, with obvious areas for character movement and interaction. Ensure the image can be tiled horizontally for seamless scrolling.",
        'Object': "Create a detailed object image for a 2D game. The object should be a key item with a transparent background, easily recognizable, and fitting the theme. Design for easy game integration and potential 3D conversion. The design should be clear, with minimal unnecessary details, and include multiple angles (front, side, top) if possible for easier 3D modeling.",
        'Texture': "Generate a seamless texture pattern suitable for use in a 2D game. The texture should be tileable and have a resolution of 512x512 pixels. Ensure the pattern is consistent and doesn't have visible seams when tiled. The texture should be versatile enough to be used on various surfaces within the game.",
        'Sprite': "Create a game sprite sheet with multiple animation frames for a character or object. The sprite should be designed with a transparent background, clear silhouette, and distinct frames for easy animation in a 2D game engine. Include at least idle, walk, and action (attack/use) animations. Ensure consistent sizing and positioning across all frames.",
        'UI': "Design a cohesive set of user interface elements for a 2D game. This should include buttons, icons, health/mana bars, and menu backgrounds. Ensure the design is clean, intuitive, and fits the game's overall aesthetic. Create with a transparent background for easy integration. The style should be consistent across all elements and scalable for different screen resolutions."
    }
    
    sizes = {
        'Character': (768, 1024),
        'Enemy': (768, 1024),
        'Background': (1024, 768),
        'Object': (1024, 1024),
        'Texture': (512, 512),
        'Sprite': (1024, 768),
        'UI': (1024, 768)
    }

    for img_type in customization['image_types']:
        for i in range(customization['image_count'].get(img_type, 0)):
            prompt = f"{image_prompts[img_type]} The design should fit the following game concept: {game_concept}. Variation {i + 1}"
            size = sizes[img_type]
            
            image_url = generate_image(prompt, size)
            
            if image_url and not isinstance(image_url, str) and not image_url.startswith('Error'):
                images[f"{img_type.lower()}_image_{i + 1}"] = image_url
                if customization['convert_to_3d'].get(img_type, False):
                    model_url = convert_image_to_3d(image_url)
                    if model_url and not isinstance(model_url, str) and not model_url.startswith('Error'):
                        images[f"{img_type.lower()}_3d_model_{i + 1}"] = model_url
            else:
                images[f"{img_type.lower()}_image_{i + 1}"] = image_url

    return images

# Clean generated code
# Clean generated code to ensure full runnable scripts only
def clean_generated_code(raw_code):
    # Remove code block markdown if present
    code = re.sub(r'```[\w]*\n|```', '', raw_code)
    
    # Remove leading intros or comments before the actual code begins
    code = re.sub(r'^[^(\nimport|\nusing|\n#include)].*?(import|using|#include|public class)', r'\1', code, flags=re.DOTALL)

    # Remove trailing content after the last code line
    code = re.sub(r'(\n\s*\n.*$)', '', code, flags=re.DOTALL)

    # Remove comments that are not part of the actual code logic
    code = re.sub(r'^\s*//.*$', '', code, flags=re.MULTILINE)

    # Remove multiline comments and instructions
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    return code.strip()

# Generate scripts based on customization settings and code types
def generate_scripts(customization, game_concept):
    scripts = {}
    code_type = 'unity' if customization['code_types']['unity'] else 'unreal' if customization['code_types']['unreal'] else 'blender'

    # Modular script descriptions based on user-selected code type
    script_descriptions = {
        'unity': {
            'Player': "Create a Unity C# player script for a 2D game with WASD controls and jump. Keep it simple without additional explanations or comments.",
            'Enemy': "Create a Unity C# enemy AI script for a 2D game. The enemy should patrol, chase the player, and attack when in range.",
            'Game Object': "Create a Unity C# script for an interactive game object that responds to player interaction and triggers animations.",
            'Level Background': "Create a Unity C# script to manage parallax scrolling for background layers in a 2D game."
        },
        'unreal': {
            'Player': "Create an Unreal C++ player character script for a 2D game with WASD movement and jump mechanics, no additional comments.",
            'Enemy': "Create an Unreal C++ enemy AI script that includes patrolling, chasing, and attacking the player when close.",
            'Game Object': "Create an Unreal C++ script for a game object that interacts with the player in a 2D environment.",
            'Level Background': "Create an Unreal C++ script for handling background parallax scrolling in a 2D game."
        },
        'blender': {
            'Player': "Create a Blender Python script to control a 3D player model with basic movement controls for a 2D game.",
            'Enemy': "Create a Blender Python script for enemy AI animations and behavior in a 2D game.",
            'Game Object': "Create a Blender Python script for interacting with objects in a 2D game, using physics-based interactions.",
            'Level Background': "Create a Blender Python script to manage background setup and animations for a 2D game environment."
        }
    }

    # Generate scripts for each type specified
    for script_type in customization['script_types']:
        for i in range(customization['script_count'].get(script_type, 0)):
            desc = script_descriptions[code_type][script_type]

            # Generate content using OpenAI or Replicate depending on code model
            if customization['code_model'] in ['gpt-4o', 'gpt-4o-mini']:
                script_code = generate_content(desc, "code generation")
            else:
                script_code = generate_content(desc, "code generation")

            # Handle truncation if needed
            full_script = handle_truncated_response(script_code, desc)

            # Ensure the entire script is saved to one file
            scripts[f"{script_type.lower()}_{code_type}_script_{i + 1}.py"] = clean_generated_code(full_script)

    return scripts



def save_script_to_file(script_name, script_content, output_dir):
    script_path = os.path.join(output_dir, f"{script_name}.py")
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)

# Generate additional game elements
def generate_additional_elements(game_concept, elements_to_generate):
    additional_elements = {}
    
    if elements_to_generate.get('storyline'):
        additional_elements['storyline'] = generate_content(f"Create a detailed storyline for the following game concept: {game_concept}. Include a compelling narrative arc, character development, and key plot points that tie into the gameplay mechanics.", "game narrative design")
    
    if elements_to_generate.get('dialogue'):
        additional_elements['dialogue'] = generate_content(f"Write sample dialogue for key characters in the following game concept: {game_concept}. Include conversations that reveal character personalities, advance the plot, and provide gameplay hints.", "game dialogue writing")
    
    if elements_to_generate.get('game_mechanics'):
        additional_elements['game_mechanics'] = generate_content(f"Describe detailed game mechanics for the following game concept: {game_concept}. Include core gameplay loops, progression systems, and unique features that set this game apart. Explain how these mechanics tie into the game's theme and story.", "game design")
    
    if elements_to_generate.get('level_design'):
        additional_elements['level_design'] = generate_content(f"Create a detailed level design document for the following game concept: {game_concept}. Include a layout sketch, key areas, enemy placement, puzzle elements, and how the level progression ties into the overall game narrative.", "game level design")
    
    return additional_elements

# Generate procedural content
def generate_procedural_world(game_concept):
    prompt = f"Create a Python script for procedurally generating a world based on this game concept: {game_concept}. Include functions for terrain generation, biome distribution, and landmark placement."
    return clean_generated_code(generate_content(prompt, "procedural world generation"))

def generate_procedural_levels(game_concept):
    prompt = f"Create a Python script for procedurally generating levels based on this game concept: {game_concept}. Include functions for room layout, obstacle placement, and difficulty scaling."
    return clean_generated_code(generate_content(prompt, "procedural level generation"))

def generate_procedural_items(game_concept):
    prompt = f"Create a Python script for procedurally generating items based on this game concept: {game_concept}. Include functions for creating weapons, armor, and consumables with varying attributes and rarity."
    return clean_generated_code(generate_content(prompt, "procedural item generation"))

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
        game_plan['game_concept'] = generate_content(f"Invent a new 2D game concept with a detailed theme, setting, and unique features based on the following prompt: {user_prompt}. Ensure the game has WASD controls and consider how it could stand out in the current market.", "game design")
    
    # Generate world concept
    if customization['generate_elements']['world_concept']:
        update_status("Creating world concept...", 0.2)
        game_plan['world_concept'] = generate_content(f"Create a detailed world concept for the 2D game: {game_plan['game_concept']}. Describe the environment, atmosphere, and any unique elements that make this world compelling for players to explore.", "world building")
    
    # Generate character concepts
    if customization['generate_elements']['character_concepts']:
        update_status("Designing characters...", 0.3)
        game_plan['character_concepts'] = generate_content(f"Create detailed character concepts for the player and enemies in the 2D game: {game_plan['game_concept']}. Include their backstories, motivations, and how they fit into the game world and mechanics.", "character design")
    
    # Generate plot
    if customization['generate_elements']['plot']:
        update_status("Crafting the plot...", 0.4)
        game_plan['plot'] = generate_content(f"Create a plot for the 2D game based on the world and characters of the game: {game_plan.get('world_concept', '')} and {game_plan.get('character_concepts', '')}. Ensure the plot integrates well with the gameplay and provides motivation for the player's actions.", "plot development")
    
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
    
    # Generate procedural content
    if customization['procedural_generation']['world']:
        update_status("Generating procedural world script...", 0.85)
        game_plan['procedural_world'] = generate_procedural_world(game_plan.get('game_concept', ''))

    if customization['procedural_generation']['levels']:
        update_status("Generating procedural level script...", 0.90)
        game_plan['procedural_levels'] = generate_procedural_levels(game_plan.get('game_concept', ''))

    if customization['procedural_generation']['items']:
        update_status("Generating procedural item script...", 0.95)
        game_plan['procedural_items'] = generate_procedural_items(game_plan.get('game_concept', ''))
    
    # Optional: Generate music
    if customization['use_replicate']['generate_music']:
        update_status("Composing background music...", 0.98)
        music_prompt = f"Create background music for the game: {game_plan.get('game_concept', '')}. The music should reflect the game's atmosphere and enhance the player's experience."
        game_plan['music'] = generate_music(music_prompt)

    update_status("Game plan generation complete!", 1.0)

    return game_plan

# Function to display images
def display_image(image_url, caption):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad responses
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=caption, use_column_width=True)
    except requests.RequestException as e:
        st.warning(f"Unable to load image: {caption}")
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.warning(f"Unable to display image: {caption}")
        st.error(f"Error: {str(e)}")

# Help and FAQ function
def show_help_and_faq():
    st.markdown("## Help & FAQ")
    
    st.markdown("### How does this app work?")
    st.write("""
    1. You input your game concept and customize settings.
    2. The app uses AI models to generate various game elements (concept, world, characters, plot, images, scripts, etc.).
    3. All generated content is compiled into a downloadable ZIP file.
    """)
    
    st.markdown("### What are the different AI models used?")
    st.markdown("""
    #### Chat Models:
    - **GPT-4**: OpenAI's most advanced language model, capable of understanding and generating human-like text.
    - **GPT-4o-mini**: A more lightweight version of GPT-4, optimized for faster responses.
    - **Llama**: An open-source large language model developed by Meta AI.

    #### Image Models:
    - **DALL-E 3**: OpenAI's advanced text-to-image generation model.
    - **SD Flux-1**: A stable diffusion model optimized for fast image generation.
    - **SDXL Lightning**: A high-speed version of Stable Diffusion XL for rapid image creation.

    #### Code Models:
    - **GPT-4o**: OpenAI's GPT-4 model optimized for code generation.
    - **GPT-4o-mini**: A lightweight version of GPT-4o for faster code generation.
    - **CodeLlama-34B**: A large language model specifically trained for code generation tasks.
    """)
    
    st.markdown("### What types of content can be generated?")
    st.write("""
    - Game concept
    - World concept
    - Character concepts
    - Plot
    - Images (characters, enemies, backgrounds, objects, textures, sprites, UI)
    - Scripts (for Unity, Unreal Engine, or Blender)
    - Additional elements like storyline, dialogue, game mechanics, and level design
    - Background music
    - Procedural generation scripts for world, levels, and items
    """)
    
    st.markdown("### How can I use the generated content?")
    st.write("""
    The generated content is meant to serve as a starting point or inspiration for your game development process. 
    You can use it as a foundation to build upon, modify, or adapt as needed for your specific game project.
    Always ensure you have the right to use AI-generated content in your jurisdiction and for your intended purpose.
    """)
    
    st.markdown("### Are there any limitations?")
    st.write("""
    - The quality and relevance of the generated content depend on the input prompts and selected AI models.
    - AI-generated content may require human review and refinement.
    - The app requires valid API keys for OpenAI and Replicate to function properly.
    - Large requests may take some time to process, depending on the selected options and server load.
    """)

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main-header {
        color: #4CAF50;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .section-header {
        color: #2196F3;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 16px;
        color: #555;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<p class="main-header">Game Dev Automation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Settings")

    # API Key Inputs
    with st.expander("API Keys"):
        openai_key = st.text_input("OpenAI API Key", value=st.session_state.api_keys['openai'], type="password")
        replicate_key = st.text_input("Replicate API Key", value=st.session_state.api_keys['replicate'], type="password")
        if st.button("Save API Keys"):
            save_api_keys(openai_key, replicate_key)
            st.session_state.api_keys['openai'] = openai_key
            st.session_state.api_keys['replicate'] = replicate_key
            st.success("API Keys saved successfully!")

    # Model Selection
    st.markdown("### AI Model Selection")
    st.session_state.customization['chat_model'] = st.selectbox(
        "Select Chat Model",
        options=['gpt-4', 'gpt-4o-mini', 'llama'],
        index=0
    )
    st.session_state.customization['image_model'] = st.selectbox(
        "Select Image Generation Model",
        options=['dall-e-3', 'SD Flux-1', 'SDXL Lightning'],
        index=0
    )
    st.session_state.customization['code_model'] = st.selectbox(
        "Select Code Generation Model",
        options=['gpt-4o', 'gpt-4o-mini', 'CodeLlama-34B'],
        index=0
    )

    if st.button("Help / FAQ"):
        show_help_and_faq()

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Game Concept", "Image Generation", "Script Generation", "Additional Elements", "Procedural Generation"])

with tab1:
    st.markdown('<p class="section-header">Define Your Game</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Describe your game concept in detail. This will be used as the foundation for generating all other elements.</p>', unsafe_allow_html=True)
    user_prompt = st.text_area("Game Concept", "Enter a detailed description of your game here...", height=200)

with tab2:
    st.markdown('<p class="section-header">Image Generation</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Customize the types and number of images you want to generate for your game.</p>', unsafe_allow_html=True)
    
    for img_type in st.session_state.customization['image_types']:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.customization['image_count'][img_type] = st.number_input(
                f"Number of {img_type} Images", 
                min_value=0, 
                value=st.session_state.customization['image_count'][img_type]
            )
        with col2:
            if img_type in ['Character', 'Enemy', 'Object', 'UI']:
                st.session_state.customization['convert_to_3d'][img_type] = st.checkbox(
                    "Make 3D",
                    value=st.session_state.customization['convert_to_3d'][img_type],
                    key=f"3d_checkbox_{img_type}"
                )

with tab3:
    st.markdown('<p class="section-header">Script Generation</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Specify the types and number of scripts you need for your game.</p>', unsafe_allow_html=True)
    
    for script_type in st.session_state.customization['script_types']:
        st.session_state.customization['script_count'][script_type] = st.number_input(
            f"Number of {script_type} Scripts", 
            min_value=0, 
            value=st.session_state.customization['script_count'][script_type]
        )

    st.markdown("### Code Type Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.customization['code_types']['unity'] = st.checkbox("Unity C# Scripts", value=st.session_state.customization['code_types']['unity'])
    with col2:
        st.session_state.customization['code_types']['unreal'] = st.checkbox("Unreal C++ Scripts", value=st.session_state.customization['code_types']['unreal'])
    with col3:
        st.session_state.customization['code_types']['blender'] = st.checkbox("Blender Python Scripts", value=st.session_state.customization['code_types']['blender'])

with tab4:
    st.markdown('<p class="section-header">Additional Game Elements</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Select additional elements to enhance your game design.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.customization['generate_elements']['storyline'] = st.checkbox("Detailed Storyline", value=st.session_state.customization['generate_elements']['storyline'])
        st.session_state.customization['generate_elements']['dialogue'] = st.checkbox("Sample Dialogue", value=st.session_state.customization['generate_elements']['dialogue'])
    with col2:
        st.session_state.customization['generate_elements']['game_mechanics'] = st.checkbox("Game Mechanics Description", value=st.session_state.customization['generate_elements']['game_mechanics'])
        st.session_state.customization['generate_elements']['level_design'] = st.checkbox("Level Design Document", value=st.session_state.customization['generate_elements']['level_design'])
    
    st.session_state.customization['use_replicate']['generate_music'] = st.checkbox("Generate Background Music", value=st.session_state.customization['use_replicate']['generate_music'])

with tab5:
    st.markdown('<p class="section-header">Procedural Generation</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Select elements to be procedurally generated:</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.customization['procedural_generation']['world'] = st.checkbox(
            "Procedural World Generation", 
            value=st.session_state.customization['procedural_generation']['world']
        )
    with col2:
        st.session_state.customization['procedural_generation']['levels'] = st.checkbox(
            "Procedural Level Generation", 
            value=st.session_state.customization['procedural_generation']['levels']
        )
    with col3:
        st.session_state.customization['procedural_generation']['items'] = st.checkbox(
            "Procedural Item Generation", 
            value=st.session_state.customization['procedural_generation']['items']
        )

# Generate Game Plan
if st.button("Generate Game Plan", key="generate_button"):
    if not st.session_state.api_keys['openai'] or not st.session_state.api_keys['replicate']:
        st.error("Please enter and save both OpenAI and Replicate API keys.")
    else:
        with st.spinner('Generating game plan...'):
            game_plan = generate_game_plan(user_prompt, st.session_state.customization)
        st.success('Game plan generated successfully!')

        # Display game plan results
        st.markdown('<p class="section-header">Generated Game Plan</p>', unsafe_allow_html=True)

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
            st.subheader("Generated Assets")
            st.write("### Images")
            for img_name, img_url in game_plan['images'].items():
                if isinstance(img_url, dict):  # This is a 3D model
                    st.write(f"{img_name}:")
                    if img_url.get('glb'):
                        st.write(f"[View 3D Model (GLB)]({img_url['glb']})")
                    if img_url.get('obj'):
                        st.write(f"[View 3D Model (OBJ)]({img_url['obj']})")
                elif isinstance(img_url, str) and not img_url.startswith('Error'):
                    display_image(img_url, img_name)
                else:
                    st.write(f"{img_name}: {img_url}")

        if 'scripts' in game_plan:
            st.write("### Scripts")
            for script_name, script_code in game_plan['scripts'].items():
                with st.expander(f"View {script_name}"):
                    st.code(script_code, language='python')

        if 'additional_elements' in game_plan:
            st.subheader("Additional Game Elements")
            for element_name, element_content in game_plan['additional_elements'].items():
                with st.expander(f"View {element_name.capitalize()}"):
                    st.write(element_content)

        if 'procedural_world' in game_plan:
            st.subheader("Procedural World Generation Script")
            with st.expander("View Procedural World Generation Script"):
                st.code(game_plan['procedural_world'], language='python')

        if 'procedural_levels' in game_plan:
            st.subheader("Procedural Level Generation Script")
            with st.expander("View Procedural Level Generation Script"):
                st.code(game_plan['procedural_levels'], language='python')

        if 'procedural_items' in game_plan:
            st.subheader("Procedural Item Generation Script")
            with st.expander("View Procedural Item Generation Script"):
                st.code(game_plan['procedural_items'], language='python')

        # Create a ZIP file with all generated content
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all generated content to files in the temporary directory
            for key, value in game_plan.items():
                if isinstance(value, str):
                    with open(os.path.join(temp_dir, f"{key}.txt"), 'w') as f:
                        f.write(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str):
                            with open(os.path.join(temp_dir, f"{key}_{sub_key}.txt"), 'w') as f:
                                f.write(sub_value)

            # Create a ZIP file
            shutil.make_archive(os.path.join(temp_dir, "game_plan"), 'zip', temp_dir)

            # Provide download button for the ZIP file
            with open(os.path.join(temp_dir, "game_plan.zip"), "rb") as f:
                st.download_button(
                    label="Download Game Plan ZIP",
                    data=f,
                    file_name="game_plan.zip",
                    mime="application/zip"
                )

        # Display generated music if applicable
        if 'music' in game_plan and game_plan['music']:
            st.subheader("Generated Music")
            st.audio(game_plan['music'], format='audio/mp3')
        else:
            st.warning("No music was generated or an error occurred during music generation.")

# Footer
st.markdown("---")
st.markdown("""
    Created by [Your Name/Company] | 
    [GitHub](https://github.com/yourusername/game-dev-automation) | 
    [Twitter](https://twitter.com/yourusername) | 
    [Website](https://www.yourwebsite.com)
    """, unsafe_allow_html=True)

# Initialize Replicate client
if st.session_state.api_keys['replicate']:
    replicate.Client(api_token=st.session_state.api_keys['replicate'])

# Main execution
if __name__ == "__main__":
    # Load API keys
    openai_key, replicate_key = load_api_keys()
    if openai_key and replicate_key:
        st.session_state.api_keys['openai'] = openai_key
        st.session_state.api_keys['replicate'] = replicate_key
