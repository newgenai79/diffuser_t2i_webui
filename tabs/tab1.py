"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os

from sana2K import sana2KInference
from cogView3Plus import CogView3PlusInference
from hunyuanDIT import HunyuanDITInference
from lumina import LuminaInference
from kandinsky3 import Kandinsky3Inference
from datetime import datetime


MAX_SEED = np.iinfo(np.int32).max
RESOLUTIONS_cogView3Plus = [
    "512x512",
    "720x480",
    "1024x1024",
    "1280x720",
    "2048x2048"
]
RESOLUTIONS_hunyuandit = [
    "1024x1024",
    "1280x1280",
    "1024x768",
    "1152x864",
    "1280x960",
    "2048x2048",
    "768x1024",
    "864x1152",
    "960x1280",
    "1280x768",
    "768x1280"
]
OUTPUT_DIR = "output"

def get_dimensions(resolution):
    width, height = map(int, resolution.split('x'))
    return width, height


def get_timestamped_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()
def generate_images(
    main_prompt, main_negative_prompt, seed,
    sana_enable, sana_prompt, sana_negative_prompt,
    sana_width, sana_height, sana_guidance_scale,
    sana_num_inference_steps, sana_memory_optimization, sana_inference_type,
    cogView3Plus_enable, cogView3Plus_memory_optimization, cogView3Plus_prompt, 
    cogView3Plus_negative_prompt, cogView3Plus_resolution, 
    cogView3Plus_guidance_scale, cogView3Plus_num_inference_steps,
    hunyuandit_enable, hunyuandit_memory_optimization, hunyuandit_prompt, 
    hunyuandit_negative_prompt, hunyuandit_resolution, 
    hunyuandit_guidance_scale, hunyuandit_num_inference_steps,
    lumina_enable, lumina_prompt, lumina_negative_prompt,
    lumina_width, lumina_height, lumina_guidance_scale,
    lumina_num_inference_steps, lumina_memory_optimization,
    kandinsky3_enable, kandinsky3_prompt, kandinsky3_negative_prompt,
    kandinsky3_width, kandinsky3_height, kandinsky3_guidance_scale,
    kandinsky3_num_inference_steps, kandinsky3_memory_optimization,
    
):
    generated_images = []
    
    # Create timestamped output directory
    output_dir = get_timestamped_output_dir()
    if kandinsky3_enable:
        prompt = kandinsky3_prompt if kandinsky3_prompt.strip() else main_prompt
        negative_prompt = kandinsky3_negative_prompt if kandinsky3_negative_prompt.strip() else main_negative_prompt
        
        try:
            image_path = Kandinsky3Inference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=kandinsky3_width,
                height=kandinsky3_height,
                guidance_scale=kandinsky3_guidance_scale,
                num_inference_steps=kandinsky3_num_inference_steps,
                seed=seed,
                optimization_mode=kandinsky3_memory_optimization,
                output_path=output_dir
            )
            generated_images.append((image_path, "Kandinsky3"))
        except Exception as e:
            print(f"Error generating Kandinsky3 image: {e}")
    if lumina_enable:
        prompt = lumina_prompt if lumina_prompt.strip() else main_prompt
        negative_prompt = lumina_negative_prompt if lumina_negative_prompt.strip() else main_negative_prompt
        
        try:
            image_path = LuminaInference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=lumina_width,
                height=lumina_height,
                guidance_scale=lumina_guidance_scale,
                num_inference_steps=lumina_num_inference_steps,
                seed=seed,
                optimization_mode=lumina_memory_optimization,
                output_path=output_dir
            )
            generated_images.append((image_path, "Lumina"))
        except Exception as e:
            print(f"Error generating Lumina image: {e}")
            
    if hunyuandit_enable:
        prompt = hunyuandit_prompt if hunyuandit_prompt.strip() else main_prompt
        negative_prompt = hunyuandit_negative_prompt if hunyuandit_negative_prompt.strip() else main_negative_prompt
        width, height = get_dimensions(hunyuandit_resolution)
        try:
            image_path = HunyuanDITInference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=hunyuandit_guidance_scale,
                num_inference_steps=hunyuandit_num_inference_steps,
                seed=seed,
                optimization_mode=hunyuandit_memory_optimization,
                output_path=output_dir
            )
            generated_images.append((image_path, "HunyuanDIT"))
        except Exception as e:
            print(f"Error generating HunyuanDIT image: {e}")

    if cogView3Plus_enable:
        prompt = cogView3Plus_prompt if cogView3Plus_prompt.strip() else main_prompt
        negative_prompt = cogView3Plus_negative_prompt if cogView3Plus_negative_prompt.strip() else main_negative_prompt
        width, height = get_dimensions(cogView3Plus_resolution)
        try:
            image_path = CogView3PlusInference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=cogView3Plus_guidance_scale,
                num_inference_steps=cogView3Plus_num_inference_steps,
                seed=seed,
                optimization_mode=cogView3Plus_memory_optimization,
                output_path=output_dir
            )
            generated_images.append((image_path, "CogView3 Plus"))
        except Exception as e:
            print(f"Error generating CogView3 Plus image: {e}")
    
    if sana_enable:
        prompt = sana_prompt if sana_prompt.strip() else main_prompt
        negative_prompt = sana_negative_prompt if sana_negative_prompt.strip() else main_negative_prompt
        
        try:
            image_path = sana2KInference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=sana_width,
                height=sana_height,
                guidance_scale=sana_guidance_scale,
                num_inference_steps=sana_num_inference_steps,
                seed=seed,
                optimization_mode=sana_memory_optimization,
                output_path=output_dir,
                inference_type=sana_inference_type,
            )
            generated_images.append((image_path, "Sana 2K"))
        except Exception as e:
            print(f"Error generating Sana image: {e}")
    

    print("Images generated: ", output_dir)
    return [(img_path, label) for img_path, label in generated_images]
def create_multi_model_tab():
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="Enter your text prompt here", 
                lines=4,
                interactive=True
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                placeholder="Enter negative prompt here",
                lines=2,
                interactive=True
            )
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            generate_button = gr.Button("Generate image(s)")
    
    output_gallery = gr.Gallery(
        label="Generated Images",
        columns=3,
        rows=1,
        height="auto"
    )

    with gr.Tabs():
        with gr.Tab("Sana 2K / Sana PAG 2K / Sana 4K"):
            with gr.Row():
                sana_enable = gr.Checkbox(label="Enable Sana", value=False, interactive=True)
                sana_inference_type = gr.Radio(
                    choices=["Sana 2K", "Sana PAG 2K", "Sana 4K"],
                    label="Inference type",
                    value="Sana 2K",
                    interactive=True
                )
                sana_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM"],
                    label="Memory Optimization",
                    value="Low VRAM",
                    interactive=True
                )
            with gr.Row():
                with gr.Column():
                    sana_prompt_input = gr.Textbox(
                        label="Prompt (Override)", 
                        placeholder="Leave empty to use main prompt", 
                        lines=3,
                        interactive=True
                    )
                    sana_negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Override)",
                        placeholder="Leave empty to use main negative prompt",
                        lines=3,
                        interactive=True
                    )
                with gr.Column():
                    with gr.Row():
                        sana_width_input = gr.Number(
                            label="Width", 
                            value=2048, 
                            minimum=512, 
                            maximum=4096, 
                            step=64,
                            interactive=True
                        )
                        sana_height_input = gr.Number(
                            label="Height", 
                            value=2048, 
                            minimum=512, 
                            maximum=4096, 
                            step=64,
                            interactive=True
                        )
                    sana_guidance_scale_slider = gr.Slider(
                        label="Guidance Scale", 
                        minimum=1.0, 
                        maximum=20.0, 
                        value=7.0, 
                        step=0.1,
                        interactive=True
                    )
                    sana_num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps", 
                        value=30,
                        interactive=True
                    )

        with gr.Tab("CogView 3 Plus"):
            with gr.Row():
                cogView3Plus_enable = gr.Checkbox(label="Enable CogView3Plus", value=False, interactive=True)
                cogView3Plus_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    value="Extremely Low VRAM",
                    interactive=True
                )
            with gr.Row():
                with gr.Column():
                    cogView3Plus_prompt_input = gr.Textbox(
                        label="Prompt (Override)",
                        placeholder="Leave empty to use main prompt",
                        lines=3,
                        interactive=True
                    )
                    cogView3Plus_negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Override)",
                        placeholder="Leave empty to use main negative prompt",
                        lines=3,
                        interactive=True
                    )
                with gr.Column():
                    cogView3Plus_resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTIONS_cogView3Plus,
                        value="512x512",
                        label="Resolution"
                    )
                    cogView3Plus_guidance_scale_slider = gr.Slider(
                        label="Guidance Scale", 
                        minimum=1.0, 
                        maximum=20.0, 
                        value=7.0, 
                        step=0.1,
                        interactive=True
                    )
                    cogView3Plus_num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps", 
                        value=30,
                        interactive=True
                    )
        with gr.Tab("Hunyuan DIT"):
            with gr.Row():
                hunyuandit_enable = gr.Checkbox(label="Enable Hunyuan DIT", value=False, interactive=True)
                hunyuandit_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM"],
                    label="Memory Optimization",
                    value="Low VRAM",
                    interactive=True
                )
            with gr.Row():
                with gr.Column():
                    hunyuandit_prompt_input = gr.Textbox(
                        label="Prompt (Override)",
                        placeholder="Leave empty to use main prompt",
                        lines=3,
                        interactive=True
                    )
                    hunyuandit_negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Override)",
                        placeholder="Leave empty to use main negative prompt",
                        lines=3,
                        interactive=True
                    )
                with gr.Column():
                    hunyuandit_resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTIONS_hunyuandit,
                        value="1280x768",
                        label="Resolution"
                    )
                    hunyuandit_guidance_scale_slider = gr.Slider(
                        label="Guidance Scale", 
                        minimum=1.0, 
                        maximum=20.0, 
                        value=5.0, 
                        step=0.1,
                        interactive=True
                    )
                    hunyuandit_num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps", 
                        value=50,
                        interactive=True
                    )
        with gr.Tab("Lumina"):
            with gr.Row():
                lumina_enable = gr.Checkbox(label="Enable Lumina", value=False, interactive=True)
                lumina_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM"],
                    label="Memory Optimization",
                    value="Low VRAM",
                    interactive=True
                )
            with gr.Row():
                with gr.Column():
                    lumina_prompt_input = gr.Textbox(
                        label="Prompt (Override)",
                        placeholder="Leave empty to use main prompt",
                        lines=3,
                        interactive=True
                    )
                    lumina_negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Override)",
                        placeholder="Leave empty to use main negative prompt",
                        lines=3,
                        interactive=True
                    )
                with gr.Column():
                    lumina_width_input = gr.Number(
                        label="Width", 
                        value=512, 
                        minimum=512, 
                        maximum=2048, 
                        step=64,
                        interactive=True
                    )
                    lumina_height_input = gr.Number(
                        label="Height", 
                        value=512, 
                        minimum=512, 
                        maximum=2048, 
                        step=64,
                        interactive=True
                    )
                    lumina_guidance_scale_slider = gr.Slider(
                        label="Guidance Scale", 
                        minimum=1.0, 
                        maximum=20.0, 
                        value=4.0, 
                        step=0.1,
                        interactive=True
                    )
                    lumina_num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps", 
                        value=30,
                        interactive=True
                    )
        with gr.Tab("Kandinsky3"):
            with gr.Row():
                kandinsky3_enable = gr.Checkbox(label="Enable Kandinsky3", value=False, interactive=True)
                kandinsky3_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    value="Extremely Low VRAM",
                    interactive=True
                )
            with gr.Row():
                with gr.Column():
                    kandinsky3_prompt_input = gr.Textbox(
                        label="Prompt (Override)",
                        placeholder="Leave empty to use main prompt",
                        lines=3,
                        interactive=True
                    )
                    kandinsky3_negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Override)",
                        placeholder="Leave empty to use main negative prompt",
                        lines=3,
                        interactive=True
                    )
                with gr.Column():
                    kandinsky3_width_input = gr.Number(
                        label="Width", 
                        value=1024, 
                        minimum=512, 
                        maximum=2048, 
                        step=64,
                        interactive=True
                    )
                    kandinsky3_height_input = gr.Number(
                        label="Height", 
                        value=1024, 
                        minimum=512, 
                        maximum=2048, 
                        step=64,
                        interactive=True
                    )
                    kandinsky3_guidance_scale_slider = gr.Slider(
                        label="Guidance Scale", 
                        minimum=1.0, 
                        maximum=20.0, 
                        value=3.0, 
                        step=0.1,
                        interactive=True
                    )
                    kandinsky3_num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps", 
                        value=25,
                        interactive=True
                    )
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])

    generate_button.click(
        fn=generate_images,
        inputs=[
            prompt_input, negative_prompt_input, seed_input,
            sana_enable, sana_prompt_input, sana_negative_prompt_input,
            sana_width_input, sana_height_input, sana_guidance_scale_slider,
            sana_num_inference_steps_input, sana_memory_optimization, sana_inference_type,
            cogView3Plus_enable, cogView3Plus_memory_optimization, cogView3Plus_prompt_input, 
            cogView3Plus_negative_prompt_input, cogView3Plus_resolution_dropdown, 
            cogView3Plus_guidance_scale_slider, cogView3Plus_num_inference_steps_input,
            hunyuandit_enable, hunyuandit_memory_optimization, hunyuandit_prompt_input, 
            hunyuandit_negative_prompt_input, hunyuandit_resolution_dropdown, 
            hunyuandit_guidance_scale_slider, hunyuandit_num_inference_steps_input,
            lumina_enable, lumina_prompt_input, lumina_negative_prompt_input,
            lumina_width_input, lumina_height_input, lumina_guidance_scale_slider,
            lumina_num_inference_steps_input, lumina_memory_optimization,
            kandinsky3_enable, kandinsky3_prompt_input, kandinsky3_negative_prompt_input,
            kandinsky3_width_input, kandinsky3_height_input, kandinsky3_guidance_scale_slider,
            kandinsky3_num_inference_steps_input, kandinsky3_memory_optimization,
        ],
        outputs=[output_gallery]
    )