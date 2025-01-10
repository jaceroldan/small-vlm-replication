import os
import gradio as gr
import asyncio
import time

from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import io
import cv2

import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "./models/smolvlm",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    device_map=DEVICE)

# Caching for gradio
# os.environ["GRADIO_TEMP_DIR"] = "./gradio_cache"
# os.environ["GRADIO_CACHE_DIR"] = "./gradio_cache"
# os.makedirs("./gradio_cache", exist_ok=True)

# processing image
app = FastAPI()
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    # Read the uploaded frame
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform inference
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[frame], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts)
    return {
        'response': generated_texts
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
