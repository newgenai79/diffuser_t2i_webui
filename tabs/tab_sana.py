"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os
from datetime import datetime
from diffusers import SanaPipeline, SanaPAGPipeline

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/Sana"

# Global variables for model caching
loaded_model_sana = None
loaded_model_type_sana = None
loaded_model_memory_mode_sana = None
loaded_model_quantization_sana = None

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(inference_type, memory_optimization, quantization):
    global loaded_model_sana, loaded_model_type_sana, loaded_model_memory_mode_sana, loaded_model_quantization_sana
    
    # If model is already loaded with same configuration, reuse it
    if (loaded_model_sana is not None and 
        loaded_model_type_sana == inference_type and 
        loaded_model_memory_mode_sana == memory_optimization and
        loaded_model_quantization_sana == quantization):
        print(f"Reusing loaded model: {inference_type}")
        return loaded_model_sana
    
    # If a different model was loaded, clear it first
    if loaded_model_sana is not None:
        del loaded_model_sana
        torch.cuda.empty_cache()
    
    pipeline_class_sana = SanaPAGPipeline if inference_type == "Sana PAG 2K" else SanaPipeline
    
    # Determine model path based on inference type
    if inference_type == "Sana 4K":
        model_path = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"
    else:
        model_path = "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers"

    # Set up quantization if enabled
    if quantization:
        print("Loading model with 8-bit quantization...")
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from transformers import BitsAndBytesConfig, AutoModel
        
        # Configure text encoder with 8-bit quantization
        text_encoder_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder = AutoModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            quantization_config=text_encoder_config,
            torch_dtype=torch.bfloat16,
        )
        
        # Configure transformer with 8-bit quantization
        transformer_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        transformer = pipeline_class_sana.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=transformer_config,
            torch_dtype=torch.bfloat16,
        ).transformer
        
        # Initialize pipeline with quantized components
        pipe_sana = pipeline_class_sana.from_pretrained(
            model_path,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
    else:
        # Common pipeline parameters for non-quantized model
        pipeline_params = {
            "variant": "bf16",
            "torch_dtype": torch.bfloat16,
            "use_safetensors": True,
            "pretrained_model_name_or_path": model_path
        }
        
        if inference_type == "Sana PAG 2K":
            pipeline_params["pag_applied_layers"] = "transformer_blocks.8"
        
        # Initialize pipeline
        pipe_sana = pipeline_class_sana.from_pretrained(**pipeline_params)
        pipe_sana.to("cuda")
        pipe_sana.vae.to(torch.bfloat16)
        pipe_sana.text_encoder.to(torch.bfloat16)
    
    print(f"Sana memory optimization mode: {memory_optimization}")
    if memory_optimization == "Low VRAM":
        pipe_sana.enable_model_cpu_offload()
        pipe_sana.enable_vae_slicing()
        
        if inference_type == "Sana 4K":
            pipe_sana.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024, tile_sample_stride_height=896, tile_sample_stride_width=896,)
        else:
            pipe_sana.enable_vae_tiling()

    if inference_type == "Sana 4K":
        if pipe_sana.transformer.config.sample_size == 128:
            from patch_conv import convert_model
            pipe_sana.vae = convert_model(pipe_sana.vae, splits=32)
    
    # Update global variables
    loaded_model_sana = pipe_sana
    loaded_model_type_sana = inference_type
    loaded_model_memory_mode_sana = memory_optimization
    loaded_model_quantization_sana = quantization
    
    return pipe_sana

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, inference_type, 
    num_images_per_prompt, quantization,
):
    # Get pipeline (either cached or newly loaded)
    pipe = get_pipeline(inference_type, memory_optimization, quantization)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Prepare inference parameters
    inference_params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "num_images_per_prompt": num_images_per_prompt,
    }
    
    # Add PAG-specific parameter if needed
    if inference_type == "Sana PAG 2K":
        inference_params["pag_scale"] = 2.0
    
    # Generate images
    images = pipe(**inference_params).images
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get base filename based on inference type
    if inference_type == "Sana 2K":
        base_filename = "sana_2k.png"
    elif inference_type == "Sana PAG 2K":
        base_filename = "sanaPAG_2k.png"
    else:
        base_filename = "sana_4k.png"
    
    # Save each image with unique timestamp and collect paths for gallery
    gallery_items = []
    for idx, image in enumerate(images):
        # Generate unique timestamp for each image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{idx+1}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        image.save(output_path)
        print(f"Image {idx+1} generated: {output_path}")
        
        # Add to gallery items
        gallery_items.append((output_path, f"{inference_type}"))
    
    return gallery_items

def create_sana_tab():
    with gr.Row():
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
        sana_quantization = gr.Checkbox(label="Enable 8bit quantization", value=False, interactive=True)
    with gr.Row():
        with gr.Column():
            sana_prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="Leave empty to use main prompt", 
                lines=3,
                interactive=True
            )
            sana_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
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
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
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
                sana_num_images_per_prompt_input = gr.Number(
                    label="Number of images/prompt", 
                    value=1,
                    minimum=1,
                    step=1,
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate image(s)")
    output_gallery = gr.Gallery(
        label="Generated Images",
        columns=3,
        rows=None,  # Allow dynamic rows based on number of images
        height="auto"
    )

    def update_dimensions(selected_type):
        return (4096, 4096) if selected_type == "Sana 4K" else (2048, 2048)
    sana_inference_type.change(update_dimensions, [sana_inference_type], [sana_width_input, sana_height_input])
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, sana_prompt_input, sana_negative_prompt_input, sana_width_input, 
            sana_height_input, sana_guidance_scale_slider, sana_num_inference_steps_input, 
            sana_memory_optimization, sana_inference_type, sana_num_images_per_prompt_input,
            sana_quantization,
        ],
        outputs=[output_gallery]
    )