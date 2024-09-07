import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image
import replicate
import base64

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
    else:
        replicate_client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
        
        try:
            if st.session_state.customization['image_model'] == 'sdxl':
                output = replicate_client.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={
                        "prompt": prompt,
                        "width": size[0],
                        "height": size[1],
                        "num_outputs": 1,
                        "scheduler": "K_EULER",
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5,
                        "prompt_strength": 0.8,
                    }
                )
            elif st.session_state.customization['image_model'] == 'midjourney':
                output = replicate_client.run(
                    "tstramer/midjourney-diffusion:436b051ebd8f68d23e83d22de5e198e0995357afef113768c20f0b6fcef23c8b",
                    input={
                        "prompt": prompt,
                        "width": size[0],
                        "height": size[1],
                        "num_outputs": 1,
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5,
                    }
                )
            else:
                return "Error: Invalid image model selected."
            
            return output[0] if output else None
        except Exception as e:
            return f"Error: Unable to generate image: {str(e)}"

# Convert image to 3D model using Replicate API
def convert_image_to_3d(image_url):
    replicate_client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
    
    try:
        prediction = replicate_client.predictions.create(
            version="d2870893aa115773465a823fe70fd446673604189843f39a99642dd9171e05e2",
            input={
                "input_image": image_url,
                "prompt": "a 3D model",
                "negative_prompt": "ugly, blurry, pixelated, obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate",
                "seed": 42
            }
        )
        
        prediction = replicate_client.predictions.wait(prediction.id)
        
        if prediction.status != "succeeded":
            st.error(f"3D conversion failed: {prediction.error}")
            return None
        
        output_urls = prediction.output
        
        result = {'glb': None, 'obj': None}
        for url in output_urls:
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
        'Player': f"Create a comprehensive player character script for a 2D game. The character should have WASD movement, jumping with spacebar, and an action button (e.g., attack or interact). Implement smooth movement, basic physics (gravity and collision), and state management (idle, walking, jumping, attacking). The player should fit the following game concept: {game_concept}. Include comments explaining each major component and potential areas for expansion.",
        'Enemy': f"Develop a detailed enemy AI script for a 2D game. The enemy should have basic pathfinding, player detection, and attack mechanics. Implement different states (idle, patrolling, chasing, attacking) and ensure smooth transitions between them. The enemy behavior should fit the following game concept: {game_concept}. Include comments explaining the AI logic and suggestions for scaling difficulty.",
        'Game Object': f"Script a versatile game object that can be used for various purposes in a 2D game. This could be a collectible item, a trap, or an interactive element. Implement functionality for player interaction, animation states, and any special effects. The object should fit the following game concept: {game_concept}. Include comments on how to easily modify the script for different object types.",
       'Level Background': f"Create a script to manage the level background in a 2D game. This should handle parallax scrolling with multiple layers, potential day/night cycles, and any interactive background elements. The background should fit the following game concept: {game_concept}. Include optimization tips and comments on how to extend the script for more complex backgrounds."
    }
    
    scripts = {}
    for script_type in customization['script_types']:
        for i in range(customization['script_count'].get(script_type, 0)):
            desc = f"{script_descriptions[script_type]} - Instance {i + 1}"
            
            if customization['code_types']['unity']:
                unity_script = generate_content(f"Create a comprehensive Unity C# script for {desc}. Include detailed comments, error handling, and optimize for performance. Ensure the script follows Unity best practices and is easily integrable into a larger project.", "Unity game development")
                scripts[f"unity_{script_type.lower()}_script_{i + 1}.cs"] = unity_script
            
            if customization['code_types']['unreal']:
                unreal_script = generate_content(f"Create a comprehensive Unreal Engine C++ script for {desc}. Include detailed comments, error handling, and optimize for performance. Ensure the script follows Unreal Engine best practices and can be easily integrated into a Blueprint system.", "Unreal Engine game development")
                scripts[f"unreal_{script_type.lower()}_script_{i + 1}.cpp"] = unreal_script
            
            if customization['code_types']['blender']:
                blender_script = generate_content(f"Create a comprehensive Blender Python script for {desc}. The script should create a detailed 3D model suitable for game development, including proper mesh topology, UV mapping, materials, and possibly animations. Include detailed comments, error handling, and tips for optimizing the model for game engines.", "Blender 3D modeling and animation")
                scripts[f"blender_{script_type.lower()}_script_{i + 1}.py"] = blender_script
    
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
    
    # Optional: Generate music
    if customization['use_replicate']['generate_music']:
        update_status("Composing background music...", 0.9)
        music_prompt = f"Create background music for the game: {game_plan.get('game_concept', '')}. The music should reflect the game's atmosphere and enhance the player's experience."
        game_plan['music'] = generate_music(music_prompt)

    update_status("Game plan generation complete!", 1.0)

    return game_plan

# Function to display images
def display_image(image_url, caption):
    st.image(image_url, caption=caption, use_column_width=True)

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
    st.image("https://your-logo-url.com/logo.png", width=100)  # Replace with your logo URL
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

    ## Model Selection
st.subheader("AI Model Selection")
st.session_state.customization['chat_model'] = st.selectbox(
    "Select Chat Model",
    options=[
        'gpt-4',
        'gpt-4o-mini',
        'llama',
        'replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1',
        'replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b',
        'replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5',
        'replicate/koala-13b:c7fc755e5ec27436b5d29923ac738427bdb2c5c03685c36d543d441b8db8fa8d'
    ],
    index=0
)

st.session_state.customization['image_model'] = st.selectbox(
    "Select Image Generation Model",
    options=[
        'dall-e-3',
        'sdxl',
        'midjourney',
        'replicate/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4',
        'replicate/kandinsky-2:ea1addaab376f4dc227f5368bbd8eff901820fd1cc14ed8cad63b29249e9d463',
        'replicate/waifu-diffusion:25d2f75ecda0c0bed34c806b7b70319a53a1bccad3ade1a7496524f013f48983',
        'replicate/openjourney:9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb'
    ],
    index=0
)

    # Replicate Options
    st.markdown("### Additional Options")
        st.session_state.customization['use_replicate']['generate_music'] = st.checkbox("Generate Music", value=st.session_state.customization['use_replicate']['generate_music'])
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Game Concept", "Asset Generation", "Script Generation", "Additional Elements"])
    
    with tab1:
        st.markdown('<p class="section-header">Define Your Game</p>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Describe your game concept in detail. This will be used as the foundation for generating all other elements.</p>', unsafe_allow_html=True)
        user_prompt = st.text_area("Game Concept", "Enter a detailed description of your game here...", height=200)
    
    with tab2:
        st.markdown('<p class="section-header">Asset Generation</p>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Customize the types and number of assets you want to generate for your game.</p>', unsafe_allow_html=True)
        
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
                        "Make 3D (feature not working)",
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
                        if isinstance(asset_url, dict):  # This is a 3D model
                            if asset_url.get('glb'):
                                glb_response = requests.get(asset_url['glb'])
                                zip_file.writestr(f"{asset_name}.glb", glb_response.content)
                            if asset_url.get('obj'):
                                obj_response = requests.get(asset_url['obj'])
                                zip_file.writestr(f"{asset_name}.obj", obj_response.content)
                        elif isinstance(asset_url, str) and asset_url.startswith('http'):
                            img_response = requests.get(asset_url)
                            img = Image.open(BytesIO(img_response.content))
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
                if 'music' in game_plan:
                    music_response = requests.get(game_plan['music'])
                    zip_file.writestr("background_music.mp3", music_response.content)
    
            st.download_button(
                "Download Game Plan ZIP",
                zip_buffer.getvalue(),
                file_name="game_plan.zip",
                mime="application/zip",
                help="Download a ZIP file containing all generated assets and documents."
            )
    
            # Display generated music if applicable
            if 'music' in game_plan:
                st.subheader("Generated Music")
                st.audio(game_plan['music'], format='audio/mp3')
    
    # Help Section
    with st.expander("Help & FAQ"):
        st.markdown("""
       
    ## Help & Frequently Asked Questions
    
    ## Basics
    
    1. **What is this tool?**  
       This is an AI-powered game development assistant that helps you create a complete game plan, including game concepts, assets, scripts, and additional elements using artificial intelligence.
    
    2. **Do I need coding skills to use this tool?**  
       No, basic game design knowledge helps, but coding is not required. However, generated scripts come with comments, so some understanding can enhance your experience.
    
    3. **What do I need to get started?**  
       You'll need a game idea and API keys for OpenAI and Replicate. You can enter them in the sidebar (to the left).
    
    ## Usage
    
    4. **How do I use this?**  
       - Start by entering your game concept under the "Game Concept" tab.
       - Customize asset generation and script generation in the relevant tabs.
       - Click 'Generate Game Plan' to create your game concept, images, scripts, and additional elements.
    
    5. **Can I edit the generated content?**  
       In the app itself, unfortunately no not yet. However, upon download of course all generated content is editable and meant to be customized to fit your specific needs.
    
    ## Game Assets
    
    6. **What game assets can this tool generate?**  
       You can generate:
       - **Images:** Characters, enemies, backgrounds, objects, textures, sprites, and UI elements.
       - **Scripts:** Code for Unity, Unreal Engine, or Blender.
       - **Music:** Background music for your game using AI music generation models.
       - **Additional Text:** Game concepts, storylines, character backgrounds, etc.
    
    7. **What image models are available?**  
       The tool supports models such as OpenAI’s DALL·E 3, SDXL, and MidJourney to create game images based on your prompts.
    
    8. **Can I create 3D models?**  
       Yes, you can convert 2D images to 3D models using Replicate's Wonder3D model. While this feature is still evolving, it provides basic 3D representations of selected assets.
    
    ## Scripts and Code
    
    9. **What kinds of scripts can this tool generate?**  
       You can generate:
       - **Player Character scripts** (WASD movement, jumping, attacking).
       - **Enemy AI scripts** (pathfinding, attack mechanics).
       - **Object interaction scripts** (collectibles, traps).
       - **Background management scripts** (parallax scrolling, day/night cycles).
       - **More**
    
    10. **What programming languages are supported?**  
       You can generate scripts in:
       - **C# for Unity**
       - **C++ for Unreal Engine**
       - **Python for Blender**
    
    ## AI Models
    
    11. **What AI models are used for generating content?**  
        - **Chat Models:** GPT-4, GPT-4o-mini, Llama 2, and many others for text generation such as game concepts and scripts.
        - **Image Models:** DALL-E 3, SDXL, Midjourney, and many others for creating visual assets.
        - **Music Models:** Replicate's MusicGen model for background music generation.
    
    ## Troubleshooting
    
    12. **What should I do if I encounter errors?**  
        - Double-check your API keys.
        - Ensure your internet connection is stable.
        - Retry generating the content, or try using different models.
    
    13. **Why does content generation take time?**  
        The time taken depends on the complexity of the content and the AI models being used. Complex game plans or large assets may take a few minutes.
     """)
    
    # Footer
    st.markdown("---")
    st.markdown("Created by Daniel Sheils.")
