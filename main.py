import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path
 
import ollama
from PIL import Image


# ---------------------------------------------------------------------------
# Prompt engineering — ask the model to act as a document analyst
# ---------------------------------------------------------------------------
 
SYSTEM_PROMPT = """\
You are a document analysis assistant. You will be shown an image of a \
document, receipt, invoice, letter, ID, note, or similar content.
 
Your job:
1. Identify what kind of document this is.
2. Extract the most important contextual details (e.g. company name, \
document type, date, person name, subject).
3. Suggest a concise, descriptive filename (without the .pdf extension).
 
Rules for the filename:
- Use underscores to separate words (e.g. Amazon_Invoice_March_2026)
- Keep it under 60 characters
- Include the most identifying details: who, what, when
- Do NOT use generic names like "Document" or "Image" or "Text_Block_1"
- Use only ASCII letters, digits, underscores, and hyphens
 
Respond ONLY with a JSON object in this exact format, no markdown fences:
{"summary": "<one-line description of the document>", "filename": "<suggested_filename>"}
"""
 
USER_PROMPT = """\
Look at this image carefully. What kind of document is this? \
Analyze its content and suggest a descriptive filename.\
"""

def encode_image(image_path:Path)->str:
    """Load an image, convert to RGB if needed, return base64 string."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def analyze_image(image_path:Path, model:str)->dict:
    """
    Send the  image to a vision model via Ollama and get back
    a structured response with a summary and suggested filename.
    """
    b64_image = encode_image(image_path)
    response = ollama.chat(
        model=model,
        messages=[
            {"role":"system", "content":SYSTEM_PROMPT},
            {
                "role":"user",
                "content":USER_PROMPT,
                "images":[b64_image]
            },
        ],
        options={"temperature":0.2} #Low temp for deterministic naming
    )

    raw = response.message.content
    if isinstance(raw,str):
        raw.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    print(f"--- Model response ---\n{raw}\n----------------------")

    #Try to parse JSON -- handle markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)

# my_img = Path("/home/arjun/Work/MyProjects/OcrProject/PaymentMade.jpg")
# res = encode_image(my_img)
# print(res)