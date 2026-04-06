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

    cleaned = str()    
    raw = response.message.content
    if isinstance(raw,str):
        raw.strip()
        print(f"--- Model response ---\n{raw}\n----------------------")
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = cleaned.strip().rstrip("`")
    else:
        print(f"Unable to process the response received from the ollama client.")
    
    # Pass 2: Ask a text-only call to generate a filename from the summary
    name_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Given this document description:\n\n\"{cleaned}\"\n\n"
                    "Generate a short, descriptive filename using underscores. "
                    "No file extension. No explanation. Just the filename.\n\n"
                    "Examples:\n"
                    "- Amazon_Invoice_March_2026\n"
                    "- John_Doe_Passport\n"
                    "- Meeting_Notes_Project_Alpha\n\n"
                    "Filename:"
                ),
            },
        ],
        options={"temperature": 0.1},
    )

    filename = name_response.message.content
    if isinstance(filename,str):
        filename = filename.strip()
        # Clean up any quotes or extra text the model might add
        filename = filename.strip('"\'').split("\n")[0].strip()

    print(f"--- Filename ---\n{filename}\n----------------")

    return {"summary": cleaned, "filename": filename}

def sanitize_filename(name:str, max_len: int = 60)->str:
    """Ensure the model-suggested filename is filesystem-safe"""
    # Strip any extension the model might have added.
    name = re.sub(r"\.(pdf|png|jpg|jpeg)$", "", name, flags=re.IGNORECASE)
    # Keep only safe characters
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe).strip("_")
    # Truncate
    safe = safe[:max_len].rstrip("_")
    return safe if safe else "Untitled_Document"




my_img = Path("/home/arjun/Work/MyProjects/OcrProject/20251001_200720.jpg")
res = analyze_image(my_img,"gemma4:e4b")
print(res["filename"])
print(sanitize_filename(res["filename"]))