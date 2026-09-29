# Virtual Fitting Room

Real-time virtual try-on kiosk for retail. Customers stand in front of a camera, browse garments and see them
on their body live, then take a snapshot to get a photorealistic render.

```
 Browser (kiosk)                                   Server (FastAPI)
 ┌───────────────────────────────────────┐        ┌───────────────────────────────┐
 │ webcam ─► MediaPipe Pose (30+ fps)    │        │ POST /api/tryon               │
 │        ─► affine-warp garment overlay │  snap  │   └─► TRYON_BACKEND           │
 │ countdown ─► capture frame ───────────┼───────►│        ├─ mock      (no key)  │
 │                                       │◄───────┼────────├─ fashn     (commercial)
 │ show photoreal result (~5-15 s)       │ image  │        └─ replicate (prototype)
 └───────────────────────────────────────┘        └───────────────────────────────┘
```

* **Live preview**: runs fully in the browser (MediaPipe Pose Landmarker, WebGL). No GPU server needed.
* **Final render**: a diffusion try-on model behind a pluggable backend.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # default TRYON_BACKEND=hf-idm is free and needs no key
.venv/bin/uvicorn backend.main:app --port 8000
```

Open http://localhost:8000 and allow camera access.

**Kiosk controls:** click a garment, or use ← / →. Space / Enter = take photo, Esc = close result.

## Backends

| `TRYON_BACKEND` | Needs | Notes |
|---|---|---|
| `hf-idm` (default) | nothing (optional `HF_TOKEN`) | **Free.** IDM-VTON on a public Hugging Face Space. Best free quality, tops only, about 20-40 s. |
| `hf-ootd` | nothing (optional `HF_TOKEN`) | **Free.** OOTDiffusion on a public HF Space. Tops, bottoms and dresses, about 15-30 s. |
| `mock` | nothing | Returns the live-overlay snapshot. Use it to test the flow offline. |
| `fashn` | `FASHN_API_KEY` | [FASHN](https://fashn.ai) try-on v1.6. **Commercial use licensed**, the recommended production option. `FASHN_MODE` trades speed for quality. |
| `replicate` | `REPLICATE_API_TOKEN` | Any Replicate model via `REPLICATE_MODEL` (default `cuuupid/idm-vton`). ⚠️ IDM-VTON / CatVTON / OOTDiffusion weights are **non-commercial**, so use them for prototyping only. |

**Free-tier limits (`hf-*`):** these Spaces run on shared ZeroGPU. Anonymous use allows only a few renders
before you hit "exceeded your ZeroGPU quota". Create a free account at https://huggingface.co, make a
read token at https://huggingface.co/settings/tokens and put it in `.env` as `HF_TOKEN=hf_...` for a larger
daily quota. Queues can be slow at peak times, and the Space owners can change or take down their Spaces.
Treat these backends as demo-only and non-commercial.

To add a backend (e.g. a self-hosted model on your own GPU), subclass `TryOnBackend` in `backend/tryon/` and
register it in `backend/tryon/__init__.py`.

## Adding real garments

The easiest way is to import a front-view product photo on a plain light background:

```bash
.venv/bin/python scripts/add_garment.py photo.jpg --id polo-white --name "Piqué Polo - White" \
    --category tops --description "white cotton piqué polo shirt"
```

This saves the photo for the try-on model, cuts out a transparent PNG for the live overlay, estimates the
anchors and adds the entry to `catalog/catalog.json`. The bundled demo garments come from the IDM-VTON Space
examples (VITON-HD research data), so replace them with your own products.

Or edit `catalog/catalog.json` by hand:

```jsonc
{
  "id": "polo-white",
  "name": "Piqué Polo - White",
  "category": "tops",                     // tops | bottoms | one-pieces
  "image": "images/polo-white.png",       // transparent PNG, front view, used for the live overlay
  "product_image": "images/polo-white.jpg", // optional: nicer photo sent to the try-on model
  "description": "white cotton piqué polo shirt",
  "overlay": { "anchors": {
    "left_shoulder":  [0.71, 0.15],       // normalised (x, y) in the image
    "right_shoulder": [0.29, 0.15],
    "mid_hip":        [0.50, 0.92]
  }}
}
```

Anchors are exactly 3 body points. The garment image is affinely mapped so those 3 points land on the matching
pose landmarks. "left" means the **wearer's** left, which is the image's right side on a front-facing photo.
Available names are `left_/right_/mid_` + `shoulder | hip | knee | ankle`, plus `nose`. Useful sets:

* tops / dresses: `left_shoulder`, `right_shoulder`, `mid_hip`
* bottoms: `left_hip`, `right_hip`, `mid_ankle` (or `mid_knee` for shorts)

## Deployment notes

* **Camera needs a secure context.** `localhost` is fine. If the kiosk browser runs on a different machine
  than the server, serve over HTTPS.
* **Privacy.** Snapshots are sent to the try-on provider. `SAVE_RESULTS=false` by default. Show a consent
  notice at the kiosk and check your data-protection obligations before enabling storage.
* **Camera placement.** Put the camera at chest height, 2-3 m away, with even front lighting and a plain
  background. Both the overlay and the diffusion models do much better when the full torso is visible.
