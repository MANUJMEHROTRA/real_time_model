import asyncio

from .base import TryOnBackend, TryOnRequest, TryOnResult


class MockBackend(TryOnBackend):
    """Echoes the live-overlay snapshot after a short delay so the kiosk flow can be tested without an API key."""

    name = "mock"

    async def run(self, req: TryOnRequest) -> TryOnResult:
        await asyncio.sleep(1.5)
        return TryOnResult(image=req.preview_image or req.person_image, backend=self.name)
