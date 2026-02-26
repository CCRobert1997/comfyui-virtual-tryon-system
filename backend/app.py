"""
FastAPI backend for ComfyUI Web App
Handles image upload, mask compositing, workflow injection, and result delivery.
"""
import copy
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from comfy_client import ComfyUIClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_PRODUCT_DIR = DATA_DIR / "input" / "product"
INPUT_BACKGROUND_DIR = DATA_DIR / "input" / "background"
OUTPUT_DIR = DATA_DIR / "output"
WORKFLOW_TEMPLATE = BASE_DIR / "workflow_template.json"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

for d in [INPUT_PRODUCT_DIR, INPUT_BACKGROUND_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ComfyUI Web App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output images at /outputs/<filename>
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

comfy = ComfyUIClient()


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_workflow_template() -> dict:
    if not WORKFLOW_TEMPLATE.exists():
        raise FileNotFoundError(
            f"workflow_template.json not found at {WORKFLOW_TEMPLATE}"
        )
    with open(WORKFLOW_TEMPLATE) as f:
        return json.load(f)


def inject_workflow(
    workflow: dict,
    product_filename: str,
    background_filename: str,
    prompt_text: str,
    output_prefix: str,
) -> dict:
    """
    Patch the workflow (deep copy):
      - node "6"  → product image  (LoadImage → ImageResize → CLIPVisionEncode → StyleModel)
      - node "1"  → background image with mask alpha (LoadImage → AddMaskForICLora)
      - node "16" → CLIPTextEncode positive prompt
      - node "39" → SaveImage (cropped final output) filename_prefix
      - node "14" → SaveImage (full image) filename_prefix
    """
    wf = copy.deepcopy(workflow)

    for node_id, node in wf.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if node_id == "6" and class_type == "LoadImage":
            inputs["image"] = product_filename
            logger.info(f"Node 6 (product) → {product_filename}")

        elif node_id == "1" and class_type == "LoadImage":
            inputs["image"] = background_filename
            logger.info(f"Node 1 (background+mask) → {background_filename}")

        elif node_id == "16" and class_type == "CLIPTextEncode":
            inputs["text"] = prompt_text
            logger.info(f"Node 16 (prompt) → {prompt_text!r}")

        elif node_id in ("39", "14") and class_type == "SaveImage":
            inputs["filename_prefix"] = output_prefix
            logger.info(f"Node {node_id} (SaveImage) prefix → {output_prefix}")

        node["inputs"] = inputs

    return wf


async def composite_mask_to_alpha(
    background_bytes: bytes, mask_bytes: bytes
) -> bytes:
    """
    Merge mask into background alpha channel for ComfyUI LoadImage.

    Frontend convention : white = modify area, black = keep area
    ComfyUI LoadImage   : MASK output = (1 - alpha/255)
                          alpha=0   → MASK=1 → inpaint HERE   (modify)
                          alpha=255 → MASK=0 → leave alone    (keep)

    So we must INVERT the mask before calling putalpha:
      frontend white (255) → inverted = 0   → alpha=0   → MASK=1 ✓ modify
      frontend black (0)   → inverted = 255 → alpha=255 → MASK=0 ✓ keep
    """
    from PIL import ImageOps

    bg = Image.open(io.BytesIO(background_bytes)).convert("RGBA")
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

    # Resize mask to match background if needed
    if mask_img.size != bg.size:
        mask_img = mask_img.resize(bg.size, Image.LANCZOS)

    # Invert: white→0, black→255
    mask_inverted = ImageOps.invert(mask_img)
    bg.putalpha(mask_inverted)

    buf = io.BytesIO()
    bg.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    product: UploadFile = File(...),
    background: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
):
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    logger.info(f"=== Generate request {run_id} ===")
    logger.info(f"Prompt: {prompt!r}")

    try:
        # ------------------------------------------------------------------
        # 1. Read uploaded files
        # ------------------------------------------------------------------
        product_bytes = await product.read()
        background_bytes = await background.read()
        mask_bytes = await mask.read()

        # ------------------------------------------------------------------
        # 2. Save originals locally
        # ------------------------------------------------------------------
        product_ext = Path(product.filename).suffix or ".png"
        background_ext = Path(background.filename).suffix or ".png"

        product_local_name = f"product_{run_id}{product_ext}"
        background_local_name = f"background_{run_id}.png"  # always PNG (RGBA)

        product_local_path = INPUT_PRODUCT_DIR / product_local_name
        background_local_path = INPUT_BACKGROUND_DIR / background_local_name

        product_local_path.write_bytes(product_bytes)

        # ------------------------------------------------------------------
        # 3. Composite mask → background alpha channel
        # ------------------------------------------------------------------
        try:
            bg_with_alpha = await composite_mask_to_alpha(background_bytes, mask_bytes)
        except Exception as e:
            logger.error(f"Mask compositing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Mask compositing error: {e}")

        background_local_path.write_bytes(bg_with_alpha)
        logger.info(f"Saved background with alpha: {background_local_path}")

        # ------------------------------------------------------------------
        # 4. Upload images to ComfyUI
        # ------------------------------------------------------------------
        try:
            product_upload = await comfy.upload_image(
                product_bytes, product_local_name, subfolder="product"
            )
            background_upload = await comfy.upload_image(
                bg_with_alpha, background_local_name, subfolder="background"
            )
        except Exception as e:
            logger.error(f"ComfyUI upload failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to upload images to ComfyUI: {e}",
            )

        # ComfyUI returns name relative to its input folder
        # When a subfolder is used, the node must receive "subfolder/name"
        def comfy_image_path(upload_result: dict) -> str:
            sub = upload_result.get("subfolder", "")
            name = upload_result["name"]
            return f"{sub}/{name}" if sub else name

        product_comfy_name = comfy_image_path(product_upload)
        background_comfy_name = comfy_image_path(background_upload)
        output_prefix = f"output_{run_id}"

        logger.info(
            f"ComfyUI paths — product: {product_comfy_name!r}, "
            f"background: {background_comfy_name!r}"
        )

        # ------------------------------------------------------------------
        # 5. Load & inject workflow
        # ------------------------------------------------------------------
        try:
            workflow = load_workflow_template()
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))

        patched_workflow = inject_workflow(
            workflow,
            product_filename=product_comfy_name,
            background_filename=background_comfy_name,
            prompt_text=prompt,
            output_prefix=output_prefix,
        )

        # ------------------------------------------------------------------
        # 6. Queue prompt
        # ------------------------------------------------------------------
        try:
            prompt_id = await comfy.queue_prompt(patched_workflow)
        except Exception as e:
            logger.error(f"Queue prompt failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to queue ComfyUI prompt: {e}",
            )

        # ------------------------------------------------------------------
        # 7. Wait for completion
        # ------------------------------------------------------------------
        try:
            history_entry = await comfy.wait_for_completion(prompt_id)
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # ------------------------------------------------------------------
        # 8. Download output images
        # ------------------------------------------------------------------
        output_images_info = comfy.extract_output_images(history_entry)
        if not output_images_info:
            raise HTTPException(
                status_code=500,
                detail="ComfyUI returned no output images.",
            )

        saved_urls = []
        for img_info in output_images_info:
            img_filename = img_info["filename"]
            img_subfolder = img_info.get("subfolder", "")
            img_type = img_info.get("type", "output")

            try:
                img_bytes = await comfy.download_image(
                    img_filename, img_subfolder, img_type
                )
            except Exception as e:
                logger.error(f"Failed to download {img_filename}: {e}")
                continue

            local_out_name = f"{run_id}_{img_filename}"
            local_out_path = OUTPUT_DIR / local_out_name
            local_out_path.write_bytes(img_bytes)
            logger.info(f"Saved output: {local_out_path}")

            saved_urls.append({"url": f"/outputs/{local_out_name}"})

        if not saved_urls:
            raise HTTPException(
                status_code=500,
                detail="Could not download any output images from ComfyUI.",
            )

        logger.info(f"=== Done {run_id}: {saved_urls} ===")
        return JSONResponse({"outputs": saved_urls})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in /generate: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
