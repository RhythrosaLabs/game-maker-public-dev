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
    else:
        return "Error: Invalid image model selected."

# Convert image to 3D model using Replicate API
def convert_image_to_3d(image_url):
    try:
        output = replicate.run(
            "adirik/wonder3d",
            input={
                "input_image": image_url
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
        'Character': "Create a highly detailed, front-facing character concept art for a 2D game...",
        'Enemy': "Design a menacing, front-facing enemy character concept art for a 2D game...",
        'Background': "Create a wide, highly detailed background image for a level of the game...",
        'Object': "Create a detailed object image for a 2D game...",
        'Texture': "Generate a seamless texture pattern...",
        'Sprite': "Create a game sprite sheet with multiple animation frames...",
        'UI': "Design a cohesive set of user interface elements for a 2D game..."
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
        'Player': f"Create a comprehensive player character script for a 2D game...",
        'Enemy': f"Develop a detailed enemy AI script for a 2D game...",
        'Game Object': f"Script a versatile game object that can be used for various purposes...",
        'Level Background': f"Create a script to manage the level background in a 2D game..."
    }
    
    scripts = {}
    selected_code_types = customization['code_types']

    for script_type in customization['script_types']:
        for i in range(customization['script_count'].get(script_type, 0)):
            desc = f"{script_descriptions[script_type]} - Instance {i + 1}"
            
            if selected_code_types['unity']:
                desc += " Generate a Unity C# script."
            if selected_code_types['unreal']:
                desc += " Generate an Unreal C++ script."
            if selected_code_types['blender']:
                desc += " Generate a Blender Python script."

            if customization['code_model'] in ['gpt-4o', 'gpt-4o-mini']:
                # Use OpenAI API for GPT-4 models
                script_code = generate_content(f"Create a comprehensive script for {desc}. Include detailed comments, error handling, and optimize for performance.", "game development")
            else:
                script_code = "Error: Invalid code model selected."

            scripts[f"{script_type.lower()}_script_{i + 1}.py"] = script_code

    return scripts

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
        game_plan['game_concept'] = generate_content(f"Invent a new 2D game concept with a detailed theme, setting, and unique features based on the following prompt: {user_prompt}.", "game design")
    
    # Generate world concept
    if customization['generate_elements']['world_concept']:
        update_status("Creating world concept...", 0.2)
        game_plan['world_concept'] = generate_content(f"Create a detailed world concept for the 2D game: {game_plan['game_concept']}.", "world building")
    
    # Generate character concepts
    if customization['generate_elements']['character_concepts']:
        update_status("Designing characters...", 0.3)
        game_plan['character_concepts'] = generate_content(f"Create detailed character concepts for the player and enemies in the 2D game: {game_plan['game_concept']}.", "character design")
    
    # Generate plot
    if customization['generate_elements']['plot']:
        update_status("Crafting the plot...", 0.4)
        game_plan['plot'] = generate_content(f"Create a plot for the 2D game.", "plot development")
    
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
        music_prompt = f"Create background music for the game."
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

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Game Concept", "Image Generation", "Script Generation", "Additional Elements"])

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
            if 'music' in game_plan and game_plan['music']:
                try:
                    music_response = requests.get(game_plan['music'])
                    music_response.raise_for_status()
                    zip_file.writestr("background_music.mp3", music_response.content)
                except requests.RequestException as e:
                    st.error(f"Error downloading music: {str(e)}")

        st.download_button(
            "Download Game Plan ZIP",
            zip_buffer.getvalue(),
            file_name="game_plan.zip",
            mime="application/zip",
            help="Download a ZIP file containing all generated assets and documents."
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
    Created by [Daniel Sheils](http://linkedin.com/in/danielsheils/) | 
    [GitHub](https://github.com/RhythrosaLabs/game-maker) | 
    [Twitter](https://twitter.com/rhythrosalabs) | 
    [Instagram](https://instagram.com/rhythrosalabs)
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
