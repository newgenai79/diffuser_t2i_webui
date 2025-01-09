"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import os
from diffusers import LuminaText2ImgPipeline

def LuminaInference(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed, optimization_mode, output_path):
    """
    Run inference using the Lumina model
    
    Args:
        prompt (str): The input prompt for image generation
        negative_prompt (str): The negative prompt for image generation
        width (int): Output image width
        height (int): Output image height
        guidance_scale (float): The guidance scale for generation
        num_inference_steps (int): Number of inference steps
        seed (int): Random seed for reproducibility
        optimization_mode (str): Memory optimization mode ("Normal" or "Low VRAM")
        output_path (str): Directory to save the generated image
        
    Returns:
        str: Path to the generated image
    """
    pipe = LuminaText2ImgPipeline.from_pretrained(
        "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    print("Lumina memory optimization mode: ", optimization_mode)
    if optimization_mode == "Low VRAM":
        pipe.enable_model_cpu_offload()
        
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )[0]
    
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, "lumina.png")
    image[0].save(filename)
    
    return filename