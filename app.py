import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, 'scripts')
sys.path.append(scripts_dir)
tabs_dir = os.path.join(current_dir, 'tabs')
sys.path.append(tabs_dir)
import gradio as gr
import torch


from tab1 import create_multi_model_tab


with gr.Blocks() as demo:
    gr.Markdown("# Text 2 Image Generation")
    with gr.Tabs():
        with gr.Tab("Multi-model text2image"):
            create_multi_model_tab()

demo.launch()