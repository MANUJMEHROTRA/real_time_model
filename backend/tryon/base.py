from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..catalog import Garment


class TryOnError(RuntimeError):
    pass


@dataclass
class TryOnRequest:
    person_image: str  # data URI of the raw (un-mirrored) camera frame
    garment: Garment
    preview_image: str | None = None  # data URI of the live-overlay composite, if the client sent one


@dataclass
class TryOnResult:
    image: str  # data URI or https URL
    backend: str


class TryOnBackend(ABC):
    name: str

    @abstractmethod
    async def run(self, req: TryOnRequest) -> TryOnResult: ...
