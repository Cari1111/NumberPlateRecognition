from dataclasses import dataclass
from .components import PreviewImage


@dataclass
class Video:
    cache_path: str
    original_path: str
    name: str


@dataclass
class Image:
    id: str  # name + number to make it unique
    cache_path: str
    name: str
    orig_file: str
    orig_fmt: str
    preview_file: str = ''
    preview_fmt: str = ''
    saved: bool = False
    has_to_be_closed: bool = False
    preview_container: PreviewImage = None

    def get_path_orig(self):
        return f'{self.cache_path}{self.id}/{self.orig_file}.{self.orig_fmt}'

    def get_path_preview(self):
        return f'{self.cache_path}{self.id}/{self.preview_file}.{self.preview_fmt}'

    def get_orig_name(self):
        return f'{self.name}.{self.orig_fmt}'

    def selected(self, selected: bool):
        if self.preview_container is None: return
        self.preview_container.toggle_selected(selected)


@dataclass
class ColorPalette:
    normal: str
    light: str
    dark: str
