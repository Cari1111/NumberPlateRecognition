from dataclasses import dataclass
from .components import PreviewImage

@dataclass
class FileVersion:
    name: str
    fmt: str

@dataclass
class FileVersionTemplate:
    name: str
    fmt: str|None = None
    max_size: int|None = None

@dataclass
class Media:
    id: str # name + number to make it unique
    cache_path: str
    name: str
    orig_file: FileVersion
    preview_file: FileVersion
    icon_file: FileVersion
    saved: bool = False
    has_to_be_closed: bool = False
    preview_container: PreviewImage = None

    def get_path_orig(self):
        return f'{self.cache_path}{self.id}/{self.orig_file.name}.{self.orig_file.fmt}'

    def get_path_preview(self):
        return f'{self.cache_path}{self.id}/{self.preview_file.name}.{self.preview_file.fmt}'

    def get_path_icon(self):
        return f'{self.cache_path}{self.id}/{self.icon_file.name}.{self.icon_file.fmt}'

    def get_orig_name(self):
        return f'{self.name}.{self.orig_file.fmt}'

    def selected(self, selected: bool):
        if self.preview_container is None: return
        self.preview_container.toggle_selected(selected)


class Image(Media):
    def __init__(self, media: Media):
        super().__init__(*media.__dict__.values())


class Video(Media):
    def __init__(self, media: Media, aspect_ratio):
        self.aspect_ratio: float = aspect_ratio
        self.position: float = 0
        super().__init__(*media.__dict__.values())

@dataclass
class ColorPalette:
    normal: str
    background: str
    dark: str
    selected: str
    text: str