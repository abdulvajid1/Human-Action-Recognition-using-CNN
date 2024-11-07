from gradio_gui import demo
from fastapi import FastAPI
import gradio as gr

app = FastAPI()

@app.get('/')
async def root():
    return "gradio app is running /gradio",200

app = gr.mount_gradio_app(app, demo, path='/gradio')
