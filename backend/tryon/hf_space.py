"""Free try-on via public Hugging Face Spaces (shared ZeroGPU). No key required; setting HF_TOKEN
(free account) raises the daily GPU quota. Both default models are NON-COMMERCIAL - demo/prototype only."""

import asyncio
import base64
import mimetypes
import tempfile
from pathlib import Path

from .. import config
from .base import TryOnBackend, TryOnError, TryOnRequest, TryOnResult

OOTD_CATEGORY = {"tops": "Upper-body", "bottoms": "Lower-body", "one-pieces": "Dress"}


class HFSpaceBackend(TryOnBackend):
    space: str

    def __init__(self) -> None:
        try:
            import gradio_client  # noqa: F401
        except ImportError as e:
            raise TryOnError("gradio_client is not installed (pip install -r requirements.txt)") from e
        self._client = None

    def _get_client(self):
        from gradio_client import Client

        if self._client is None:
            self._client = Client(self.space, token=config.HF_TOKEN or None, verbose=False)
        return self._client

    def _predict(self, person: Path, garment: Path, req: TryOnRequest) -> str:
        raise NotImplementedError

    async def run(self, req: TryOnRequest) -> TryOnResult:
        with tempfile.TemporaryDirectory() as tmp:
            person = Path(tmp) / "person.jpg"
            person.write_bytes(base64.b64decode(req.person_image.split(",", 1)[1]))
            try:
                out = await asyncio.wait_for(
                    asyncio.to_thread(self._predict, person, req.garment.product_path, req),
                    timeout=config.TRYON_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise TryOnError(f"{self.space} timed out (the free queue may be busy)")
            except TryOnError:
                raise
            except Exception as e:
                msg = str(e)
                if "quota" in msg.lower():
                    msg += " - set HF_TOKEN (free Hugging Face account) for a larger quota"
                raise TryOnError(f"{self.space}: {msg}") from e
        mime = mimetypes.guess_type(out)[0] or "image/png"
        return TryOnResult(image=f"data:{mime};base64,{base64.b64encode(Path(out).read_bytes()).decode()}",
                           backend=self.name)


class IDMVTONSpace(HFSpaceBackend):
    """IDM-VTON (yisol/IDM-VTON). Best quality of the free options; ~20-40 s. Tops only in the Space."""

    name = "hf-idm"
    space = "yisol/IDM-VTON"

    def _predict(self, person, garment, req):
        from gradio_client import handle_file

        if req.garment.category != "tops":
            raise TryOnError("The IDM-VTON Space only supports tops - use hf-ootd for bottoms/dresses")
        out, _mask = self._get_client().predict(
            dict={"background": handle_file(str(person)), "layers": [], "composite": None},
            garm_img=handle_file(str(garment)),
            garment_des=req.garment.description,
            is_checked=True,  # auto-mask
            is_checked_crop=True,  # crop/resize to the model's 3:4 input, paste result back
            denoise_steps=30,
            seed=42,
            api_name="/tryon",
        )
        return out


class OOTDiffusionSpace(HFSpaceBackend):
    """OOTDiffusion (levihsu/OOTDiffusion). Supports tops, bottoms and dresses; ~15-30 s."""

    name = "hf-ootd"
    space = "levihsu/OOTDiffusion"

    def _predict(self, person, garment, req):
        from gradio_client import handle_file

        common = dict(vton_img=handle_file(str(person)), garm_img=handle_file(str(garment)),
                      n_samples=1, n_steps=20, image_scale=2, seed=-1)
        if req.garment.category == "tops":
            gallery = self._get_client().predict(**common, api_name="/process_hd")
        else:
            gallery = self._get_client().predict(**common, category=OOTD_CATEGORY[req.garment.category],
                                                 api_name="/process_dc")
        return gallery[0]["image"]
