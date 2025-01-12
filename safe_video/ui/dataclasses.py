from dataclasses import dataclass, field
from enum import Flag, auto


@dataclass
class FileVersion:
    name: str
    fmt: str


@dataclass
class FileVersionTemplate:
    name: str
    fmt: str | None = None
    max_size: int | None = None

class Version(Flag):
    ORIG = auto()
    PREVIEW = auto()
    ICON = auto()
    ORIG_CENSORED = auto()
    PREVIEW_CENSORED = auto()


@dataclass
class Media:
    id: str  # name + number to make it unique
    cache_path: str
    name: str
    files: dict[Version, FileVersion] = field(default_factory=lambda: {
        Version.ORIG: FileVersion(name='original', fmt=None),
        Version.PREVIEW: FileVersion(name='preview', fmt='webp'),
        Version.ICON: FileVersion(name='icon', fmt='webp'),
        Version.ORIG_CENSORED: FileVersion(name='original_censored', fmt=None),
        Version.PREVIEW_CENSORED: FileVersion(name='preview_censored', fmt='webp'),
    })
    saved: bool = False
    has_to_be_closed: bool = False
    censored_available: bool = False
    preview_container = None

    def set_file(self, version: Version, file: FileVersion):
        self.files[version] = file

    def set_orig_fmt(self, fmt):
        self.files[Version.ORIG].fmt = fmt
        self.files[Version.ORIG_CENSORED].fmt = fmt

    def get_path(self, version: Version):
        return f'{self.cache_path}{self.id}/{self.files[version].name}.{self.files[version].fmt}'

    def get_path_preview(self, censored_if_available):
        return self.get_path(Version.PREVIEW_CENSORED if censored_if_available and self.censored_available else Version.PREVIEW)

    def get_orig_name(self):
        return f'{self.name}.{self.files[Version.ORIG].fmt}'

    def selected(self, selected: bool):
        if self.preview_container is None:
            return
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
