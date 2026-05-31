#!/usr/bin/env python3
"""WebSocket policy server for evaluating SimVLA on RoboCasa365."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import websockets
from safetensors.torch import load_file as load_safetensors

try:
    import msgpack
    import msgpack_numpy

    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.processing_smolvlm_vla import SmolVLMVLAProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_smolvlm_robocasa")

model: SmolVLMVLA | None = None
processor: SmolVLMVLAProcessor | None = None
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "state_dim": 16,
    "action_dim": 12,
    "sampling_steps": 10,
}


def _decode_numpy(obj):
    if isinstance(obj, dict) and (b"__ndarray__" in obj or "__ndarray__" in obj):
        data_key = b"data" if b"data" in obj else "data"
        dtype_key = b"dtype" if b"dtype" in obj else "dtype"
        shape_key = b"shape" if b"shape" in obj else "shape"
        dtype = obj[dtype_key]
        if isinstance(dtype, bytes):
            dtype = dtype.decode()
        return np.frombuffer(obj[data_key], dtype=np.dtype(dtype)).reshape(tuple(obj[shape_key]))
    return obj


def _load_model(checkpoint: str, norm_stats: str | None, smolvlm_model: str):
    global model, processor

    logger.info("Loading SimVLA checkpoint: %s", checkpoint)
    config = SmolVLMVLAConfig.from_pretrained(checkpoint)
    if smolvlm_model:
        config.smolvlm_model_path = smolvlm_model

    # Avoid nested from_pretrained calls inside Transformers' meta-device
    # loading context. SmolVLMVLA.__init__ itself loads the VLM backbone, so we
    # instantiate normally and then restore the saved SimVLA weights by hand.
    model = SmolVLMVLA(config)
    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    torch_path = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        state_dict = load_safetensors(safetensors_path, device="cpu")
    elif os.path.exists(torch_path):
        state_dict = torch.load(torch_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find model.safetensors or pytorch_model.bin in {checkpoint}"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys while loading checkpoint: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys while loading checkpoint: %d", len(unexpected))
    model = model.to(device).eval()
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model)

    if norm_stats:
        logger.info("Loading norm stats: %s", norm_stats)
        model.action_space.load_norm_stats(norm_stats)

    CONFIG["action_horizon"] = int(model.num_actions)
    CONFIG["image_size"] = int(model.image_size)
    logger.info("Loaded model on %s, action_horizon=%s", device, model.num_actions)


def _infer(request: dict[str, Any]) -> dict[str, Any]:
    assert model is not None
    assert processor is not None

    image_left = _decode_numpy(request["observation/image"])
    image_wrist = _decode_numpy(request["observation/wrist_image"])
    image_right = _decode_numpy(request.get("observation/right_image", np.zeros_like(image_left)))
    state = _decode_numpy(request.get("observation/state", np.zeros(CONFIG["state_dim"], dtype=np.float32)))
    prompt = str(request.get("prompt", ""))

    image_inputs = processor.encode_image([image_left, image_wrist, image_right])
    lang = processor.encode_language([prompt])

    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] < CONFIG["state_dim"]:
        state = np.pad(state, (0, CONFIG["state_dim"] - state.shape[-1]))
    state = state[: CONFIG["state_dim"]]

    inputs = {
        "input_ids": lang["input_ids"].to(device),
        "image_input": image_inputs["image_input"].to(device),
        "image_mask": image_inputs["image_mask"].to(device),
        "proprio": torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
    }

    with torch.no_grad():
        actions = model.generate_actions(**inputs, steps=CONFIG["sampling_steps"])
    return {"actions": actions.squeeze(0).detach().cpu().numpy()}


async def _handle_connection(websocket, path=None):
    metadata = {
        "model": "SimVLA-RoboCasa",
        "action_dim": CONFIG["action_dim"],
        "state_dim": CONFIG["state_dim"],
        "action_horizon": CONFIG.get("action_horizon", None),
    }
    if HAS_MSGPACK:
        await websocket.send(msgpack_numpy.packb(metadata, use_bin_type=True))
    else:
        import json

        await websocket.send(json.dumps(metadata))

    async for message in websocket:
        try:
            if HAS_MSGPACK and isinstance(message, bytes):
                request = msgpack_numpy.unpackb(message, raw=False)
            else:
                import json

                request = json.loads(message)

            result = _infer(request)
            actions = result["actions"].tolist()
            response = {"actions": actions}
            if HAS_MSGPACK:
                await websocket.send(msgpack.packb(response, use_bin_type=True))
            else:
                import json

                await websocket.send(json.dumps(response))
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            traceback.print_exc()
            fallback = {"actions": np.zeros((CONFIG.get("action_horizon", 16), CONFIG["action_dim"])).tolist()}
            if HAS_MSGPACK:
                await websocket.send(msgpack.packb(fallback, use_bin_type=True))
            else:
                import json

                await websocket.send(json.dumps(fallback))


async def _serve(host: str, port: int):
    async with websockets.serve(_handle_connection, host, port, max_size=None, compression=None):
        logger.info("SimVLA RoboCasa server listening on %s:%d", host, port)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, default=None)
    parser.add_argument("--smolvlm_model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not HAS_MSGPACK:
        logger.warning("Install msgpack-numpy for best compatibility with the eval client.")

    CONFIG["sampling_steps"] = args.sampling_steps
    _load_model(args.checkpoint, args.norm_stats, args.smolvlm_model)
    asyncio.run(_serve(args.host, args.port))


if __name__ == "__main__":
    main()
