#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import string
import struct
from pathlib import Path

import httpx
import msgpack


SIZES = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}


def parse_stream(blob: bytes) -> list[dict]:
    out: list[dict] = []
    i = 0
    n = len(blob)
    while i + 4 <= n:
        size = struct.unpack(">I", blob[i : i + 4])[0]
        i += 4
        if size <= 0 or i + size > n:
            break
        payload = blob[i : i + size]
        i += size
        try:
            obj = msgpack.unpackb(payload, raw=False)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Local smoke test for NovelAI image stream.")
    p.add_argument("--token", required=True, help="NovelAI access token (without 'Bearer ')")
    p.add_argument("--image-base", default="https://image.novelai.net")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")
    p.add_argument("--model", default="nai-diffusion-4-5-full")
    p.add_argument("--size", choices=["portrait", "landscape", "square"], default="square")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--scale", type=int, default=5)
    p.add_argument("--sampler", default="k_euler_ancestral")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--save-dir", default=".")
    p.add_argument("--recaptcha-token", default="")
    p.add_argument("--shared-trial", action="store_true", default=False)
    args = p.parse_args()

    w, h = SIZES[args.size]
    payload = {
        "input": args.prompt,
        "model": args.model,
        "action": "generate",
        "use_new_shared_trial": bool(args.shared_trial),
        "parameters": {
            "stream": "msgpack",
            "params_version": 3,
            "image_format": "png",
            "width": w,
            "height": h,
            "steps": args.steps,
            "scale": args.scale,
            "sampler": args.sampler,
            "n_samples": args.n,
            "v4_prompt": {"caption": {"base_caption": args.prompt, "char_captions": []}, "use_coords": False, "use_order": True},
        },
    }
    payload["parameters"]["negative_prompt"] = args.negative or ""
    payload["parameters"]["uc"] = args.negative or ""
    payload["parameters"]["v4_negative_prompt"] = {
        "caption": {"base_caption": args.negative or "", "char_captions": []},
        "legacy_uc": False,
    }
    if args.recaptcha_token:
        payload["recaptcha_token"] = args.recaptcha_token

    headers = {"Authorization": f"Bearer {args.token}", "Accept": "*/*"}
    headers["Referer"] = "https://novelai.net/"
    headers["x-correlation-id"] = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
    url = args.image_base.rstrip("/") + "/ai/generate-image-stream"

    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        files = [("request", (None, json.dumps(payload, ensure_ascii=False), "application/json"))]
        resp = client.post(url, files=files, headers=headers)

    print(f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        print(resp.text[:2000])
        return 1

    events = parse_stream(resp.content)
    print(f"events={len(events)}")
    final_images = []
    fallback_images = []
    for e in events:
        et = str(e.get("event_type", ""))
        msg = str(e.get("message", ""))
        code = str(e.get("code", ""))
        if et in {"error", "retry"}:
            print(f"{et}: code={code} msg={msg}")
        img = e.get("image")
        if isinstance(img, (bytes, bytearray)) and img:
            raw = bytes(img)
            fallback_images.append(raw)
            if et == "final":
                final_images.append(raw)

    images = final_images if final_images else fallback_images
    if not images:
        print("No image bytes found in stream events.")
        print(json.dumps({"model": args.model, "size": f"{w}x{h}", "steps": args.steps, "scale": args.scale}, ensure_ascii=False))
        return 2

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "nai_smoke_final.png"
    out.write_bytes(images[-1])
    print(f"saved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
