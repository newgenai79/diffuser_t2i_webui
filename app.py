"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import sys
import os
import gradio as gr
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
tabs_dir = os.path.join(current_dir, 'tabs')
sys.path.append(tabs_dir)

from tab_sana import create_sana_tab

with gr.Blocks() as demo:
    gr.Markdown("# Text 2 Image Generation using Diffusers")
    with gr.Tabs():
        with gr.Tab("Sana"):
            create_sana_tab()

demo.launch(share=False)