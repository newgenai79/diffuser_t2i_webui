<h1 align='center'>Diffusers text to image webui</h1>

<div align='center'>
Credits: <a href='https://github.com/huggingface/diffusers' target='_blank'>Huggingface/Diffusers team</a>
</div>
<br>
<div align='center'>
    <a href='https://github.com/NVlabs/Sana'><img src='https://img.shields.io/badge/Sana-blue'></a>
	<a href='https://github.com/NVlabs/Sana'><img src='https://img.shields.io/badge/Sana_PAG-red'></a>
	<a href='https://github.com/THUDM/CogView3'><img src='https://img.shields.io/badge/CogView_3_Plus-blue'></a>
	<a href='https://github.com/Tencent/HunyuanDiT'><img src='https://img.shields.io/badge/HunyuanDIT-red'></a>
	<a href='https://github.com/ai-forever/Kandinsky-3'><img src='https://img.shields.io/badge/Kandinsky3-blue'></a>
	<a href='https://github.com/Alpha-VLLM/Lumina-mGPT'><img src='https://img.shields.io/badge/Lumina-red'></a>
</div>

<h3 align='center'> Installation steps</h3>

<b>Step 1: Clone the repository</b>
```	
git clone https://github.com/newgenai79/diffuser_t2i_webui
```

<b>Step 2: Navigate inside the cloned repository</b>
```	
cd diffuser_t2i_webui
```

<b>Step 3: Create virtual environment</b>
```	
conda create -n dt2i python==3.10.11 -y
```

<b>Step 4: Activate virtual environment</b>
```	
conda activate dt2i
```

<b>Step 5: Install requirements</b>

Windows ```pip install -r requirements.txt```
Linux ```pip3 install -r requirements.txt```

	
<b>Step 7: Launch gradio based WebUI</b>
```	
python app.py
```