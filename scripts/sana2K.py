"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import os
from diffusers import SanaPipeline, SanaPAGPipeline

def sana2KInference(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed, optimization_mode, output_path, inference_type):

    # Select appropriate pipeline based on PAG mode
    pipeline_class = SanaPAGPipeline if inference_type == "Sana PAG 2K" else SanaPipeline
    
    # Common pipeline parameters
    pipeline_params = {
        
        "variant": "bf16",
        "torch_dtype": torch.bfloat16,
        "use_safetensors": True,
    }
    
    # Add PAG-specific parameters if needed "Sana 2K", "Sana PAG 2K", "Sana 4K"
       
    if inference_type == "Sana 4K":
        pipeline_params["pretrained_model_name_or_path"] = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"
    else:
        pipeline_params["pretrained_model_name_or_path"] = "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers"

    if inference_type == "Sana PAG 2K":
        pipeline_params["pag_applied_layers"] = "transformer_blocks.8"
    
    # Initialize pipeline
    pipe = pipeline_class.from_pretrained(**pipeline_params)
    pipe.to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    
    print("Sana 2K memory optimization mode: ", optimization_mode)
    if optimization_mode == "Low VRAM":
        pipe.enable_model_cpu_offload()

    if inference_type == "Sana 4K":
        if pipe.transformer.config.sample_size == 128:
            from patch_conv import convert_model
            pipe.vae = convert_model(pipe.vae, splits=32)
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
    }
    
    # Add PAG-specific parameter if needed
    if inference_type == "Sana PAG 2K":
        inference_params["pag_scale"] = 2.0
    
    # Generate image
    image = pipe(**inference_params)[0]
    
    # Save image
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, "sana_2k.png" if inference_type == "Sana 2K" else "sanaPAG_2k.png" if inference_type == "Sana PAG 2K" else "sana_4k.png")
    image[0].save(filename)
    
    return filename