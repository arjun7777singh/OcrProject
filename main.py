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

def convert_image_to_pdf(
        image_path: Path, model: str, max_filename_len: int = 60
)->Path:
    """
    Open an image, analyze it with a vision model, and save as a context-aware named PDF
    """
    print(f"Analyzing image with model '{model}'...")

    #Step 1: Vision model analyzes the image
    analysis = analyze_image(image_path, model)
    summary = analysis.get("summary", "")
    suggested_name = analysis.get("filename", "Untitled_Document")

    print(f"Summary : {summary}")
    print(f"Filename: {suggested_name}")

    #Step 2: Sanitize the suggested filename
    safe_name = sanitize_filename(suggested_name, max_len=max_filename_len)
    output_path = image_path.parent / f"{safe_name}.pdf"

    # Avoid overwriting
    counter = 1
    original  = output_path
    while output_path.exists():
        output_path = original.with_stem(f"{original.stem}_{counter}")
        counter+=1

    # Step 3: Save image as PDF
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(output_path, "PDF", resolution = 150.0)

    print(f"PDF saved -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Convert image to a context-aware named PDF using a vision model."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--model",
        type=str,
        default="glm-ocr:latest",
        help="Ollama vision model to use (default:glm-ocr:latest)"
        "Options: glm-ocr:latest, llama3.2-vision:latest, gemma4:e4b"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=60,
        help="Max characters for the generated filename (default: 60)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: file not found -> {args.image}", file=sys.stderr)
        sys.exit(1)
    convert_image_to_pdf(args.image, model=args.model, max_filename_len=args.max_len)

if __name__ == "__main__":
    main()


# my_img = Path("/home/arjun/Work/MyProjects/OcrProject/20251001_200720.jpg")
# res = analyze_image(my_img,"gemma4:e4b")
# print(res["filename"])
# print(sanitize_filename(res["filename"]))