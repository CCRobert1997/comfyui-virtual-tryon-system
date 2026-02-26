"""
ComfyUI API Client
Handles communication with ComfyUI server.
"""
import asyncio
import httpx
import logging
import json
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

COMFY_URL = "http://127.0.0.1:8188"
POLL_INTERVAL = 1.5   # seconds between history polls
MAX_WAIT = 300        # max seconds to wait for generation


class ComfyUIClient:
    def __init__(self, base_url: str = COMFY_URL):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Upload image to ComfyUI input folder
    # ------------------------------------------------------------------
    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> dict:
        """Upload an image to ComfyUI and return the server-side filename."""
        async with httpx.AsyncClient(timeout=60) as client:
            files = {"image": (filename, image_bytes, "image/png")}
            data = {
                "subfolder": subfolder,
                "type": "input",
                "overwrite": "true" if overwrite else "false",
            }
            resp = await client.post(
                f"{self.base_url}/upload/image", files=files, data=data
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Uploaded image: {result}")
            return result  # {"name": ..., "subfolder": ..., "type": ...}

    # ------------------------------------------------------------------
    # Queue a prompt
    # ------------------------------------------------------------------
    async def queue_prompt(self, workflow: dict, client_id: str = "comfyui-web") -> str:
        """Submit workflow to ComfyUI queue. Returns prompt_id."""
        payload = {"prompt": workflow, "client_id": client_id}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/prompt", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            prompt_id = data["prompt_id"]
            logger.info(f"Queued prompt: {prompt_id}")
            return prompt_id

    # ------------------------------------------------------------------
    # Poll history until done
    # ------------------------------------------------------------------
    async def wait_for_completion(self, prompt_id: str) -> dict:
        """Poll /history/{prompt_id} until execution completes."""
        start = time.time()
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                if time.time() - start > MAX_WAIT:
                    raise TimeoutError(
                        f"ComfyUI did not finish within {MAX_WAIT}s"
                    )
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    resp = await client.get(
                        f"{self.base_url}/history/{prompt_id}"
                    )
                    resp.raise_for_status()
                    history = resp.json()
                except httpx.HTTPStatusError as e:
                    logger.warning(f"History poll error: {e}")
                    continue

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "success" or (
                        "outputs" in entry and entry["outputs"]
                    ):
                        logger.info(f"Prompt {prompt_id} completed.")
                        return entry
                    if status.get("status_str") == "error":
                        msgs = status.get("messages", [])
                        raise RuntimeError(
                            f"ComfyUI execution error: {msgs}"
                        )

    # ------------------------------------------------------------------
    # Download output image bytes
    # ------------------------------------------------------------------
    async def download_image(
        self, filename: str, subfolder: str = "", image_type: str = "output"
    ) -> bytes:
        """Download a generated image from ComfyUI /view endpoint."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(f"{self.base_url}/view", params=params)
            resp.raise_for_status()
            return resp.content

    # ------------------------------------------------------------------
    # Extract output images from history entry
    # ------------------------------------------------------------------
    def extract_output_images(self, history_entry: dict) -> list[dict]:
        """Return list of {filename, subfolder, type} from history outputs."""
        images = []
        outputs = history_entry.get("outputs", {})
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                images.append(img)
        return images
