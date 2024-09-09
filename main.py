import streamlit as st
import requests
import json
import os
import zipfile
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
    if st.session_state.customization['chat_model'] in ['gpt-4o', 'gpt-4o-mini']:
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
def generate_image(prompt, size, steps=25, guidance=3.0, interval=2.0):
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
            # Convert size to aspect ratio
            width, height = size
            if width == height:
                aspect_ratio = "1:1"
            elif width > height:
                aspect_ratio = "16:9" if width / height > 1.7 else "3:2"
            else:
                aspect_ratio = "9:16" if height / width > 1.7 else "2:3"

            # Initialize Replicate client with API key
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])

            output = client.run(
                "black-forest-labs/flux-pro",
                input={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "steps": steps,
                    "guidance": guidance,
                    "interval": interval,
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
        output = replicate.run(
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
                    if model_url and not model_url.startswith('Error'):
                        images[f"{img_type.lower()}_3d_model_{i + 1}"] = model_url
            else:
                images[f"{img_type.lower()}_image_{i + 1}"] = image_url

    return images

# Generate scripts based on customization settings and code types
def generate_scripts(customization, game_concept):
    script_descriptions = {
        'Player': f"Create a comprehensive player character script for a Unity 2D game. The character should have WASD movement, jumping with spacebar, and an action button (e.g., attack or interact). Implement smooth movement, basic physics (gravity and collision), and state management (idle, walking, jumping, attacking). The player should fit the following game concept: {game_concept}. Include comments explaining each major component and potential areas for expansion.",
        'Enemy': f"Develop a detailed enemy AI script for a Unity 2D game. The enemy should have basic pathfinding, player detection, and attack mechanics. Implement different states (idle, patrolling, chasing, attacking) and ensure smooth transitions between them. The enemy behavior should fit the following game concept: {game_concept}. Include comments explaining the AI logic and suggestions for scaling difficulty.",
        'Game Object': f"Script a versatile game object that can be used for various purposes in a Unity 2D game. This could be a collectible item, a trap, or an interactive element. Implement functionality for player interaction, animation states, and any special effects. The object should fit the following game concept: {game_concept}. Include comments on how to easily modify the script for different object types.",
        'Level Background': f"Create a script to manage the level background in a Unity 2D game. This should handle parallax scrolling with multiple layers, potential day/night cycles, and any interactive background elements. The background should fit the following game concept: {game_concept}. Include optimization tips and comments on how to extend the script for more complex backgrounds."
    }
    
    scripts = {}
    for script_type in customization['script_types']:
        for i in range(customization['script_count'].get(script_type, 0)):
            desc = f"{script_descriptions[script_type]} - Instance {i + 1}"
            
            if customization['code_model'] in ['gpt-4o', 'gpt-4o-mini']:
                script_code = generate_content(f"Create a comprehensive Unity C# script for {desc}. Include detailed comments, error handling, and optimize for performance.", "Unity game development")
            elif customization['code_model'] == 'CodeLlama-34B':
                try:
                    client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
                    output = client.run(
                        "andreasjansson/codellama-34b-instruct-gguf:97a1fb465d5cdf2854c89ebeaee3ceb353206b8187b665a83bcf6efd21e534ab",
                        input={
                            "prompt": f"Create a comprehensive Unity C# script for {desc}. Include detailed comments, error handling, and optimize for performance.",
                            "grammar": "root        ::= \"```csharp\\n\" code \"```\"\ncode        ::= [^`]+",
                            "jsonschema": ""
                        }
                    )
                    script_code = ''.join(output)
                except Exception as e:
                    script_code = f"Error generating script: {str(e)}"
            else:
                script_code = "Error: Invalid code model selected."

            scripts[f"{script_type.lower()}_script_{i + 1}.cs"] = script_code

    return scripts

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

# Function to handle API calls and file saving
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

    # Optional: Generate music
    if customization['use_replicate']['generate_music']:
        update_status("Composing background music...", 0.9)
        music_prompt = f"Create background music for the game: {game_plan.get('game_concept', '')}. The music should reflect the game's atmosphere and enhance the player's experience."
        game_plan['music'] = generate_music(music_prompt)

    update_status("Game plan generation complete!", 1.0)

    return game_plan

# Sidebar for settings and options
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
        "Select Text Model",
        options=['gpt-4o', 'gpt-4o-mini', 'llama'],
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

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Game Concept", "Image Generation", "Script Generation", "Additional Elements"])

with tab1:
    st.markdown("### Define Your Game")
    user_prompt = st.text_area("Game Concept", "Enter a detailed description of your game here...", height=200)

with tab2:
    st.markdown("### Image Generation")
    
    for img_type in st.session_state.customization['image_types']:
        st.session_state.customization['image_count'][img_type] = st.number_input(
            f"Number of {img_type} Images", 
            min_value=0, 
            value=st.session_state.customization['image_count'][img_type]
        )
        if img_type in ['Character', 'Enemy', 'Object', 'UI']:
            st.session_state.customization['convert_to_3d'][img_type] = st.checkbox(
                "Make 3D", value=st.session_state.customization['convert_to_3d'][img_type])

with tab3:
    st.markdown("### Script Generation")
    
    for script_type in st.session_state.customization['script_types']:
        st.session_state.customization['script_count'][script_type] = st.number_input(
            f"Number of {script_type} Scripts", 
            min_value=0, 
            value=st.session_state.customization['script_count'][script_type]
        )

with tab4:
    st.markdown("### Additional Game Elements")
    
    st.session_state.customization['generate_elements']['storyline'] = st.checkbox("Detailed Storyline", value=st.session_state.customization['generate_elements']['storyline'])
    st.session_state.customization['generate_elements']['dialogue'] = st.checkbox("Sample Dialogue", value=st.session_state.customization['generate_elements']['dialogue'])

# Generate Game Plan Button
if st.button("Generate Game Plan"):
    if not st.session_state.api_keys['openai'] or not st.session_state.api_keys['replicate']:
        st.error("Please enter and save both OpenAI and Replicate API keys.")
    else:
        with st.spinner('Generating game plan...'):
            game_plan = generate_game_plan(user_prompt, st.session_state.customization)
        st.success('Game plan generated successfully!')
        
        # Display the generated content here...
