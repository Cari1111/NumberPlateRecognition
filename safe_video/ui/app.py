import flet as ft
from safe_video.number_plate_recognition import NumberPlateRecognition
from .dataclasses import Video, Image, ColorPalette
from .components import PreviewImage, AlertSaveWindow, VideoPlayer
from .helper_classes import FileManger

DarkColors = ColorPalette(
    normal = "#1a1e26",
    background = "#232833",
    dark = "#101217",
    selected = '#2b84ff',
    text =  '#aec1eb'
)

class UI_App:
    def __init__(self):
        self.colors: ColorPalette = DarkColors
        self.file_manager = FileManger(self.colors)
        self.page: ft.Page = None
        self.media_container = ft.Container(expand=True, image_fit=ft.ImageFit.CONTAIN, margin=10)
        self.preview_bar = ft.ListView([], expand=True, spacing=10)
        self.selected_img: str = None
        self.selected = set()
        self.npr = NumberPlateRecognition()
        self.file_picker_open = ft.FilePicker(on_result=self.upload_callback)
        self.file_picker_export = ft.FilePicker(on_result=self.export_callback)

    def blur_callback(self):
        self.npr.blur_image()

    def upload_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.files is None or len(file_results.files) == 0: return
        ids = []
        for file in file_results.files:
            id = self.file_manager.upload_media(old_path=file.path, filename=file.name)
            if id is not None: ids.append(id)
        self.load_images(ids)

    def load_images(self, ids: list[str]):
        if len(ids) == 0: return
        for id in ids:
            media = self.file_manager[id]
            media.preview_container = PreviewImage(media.id, media.get_path_icon(), self.switch_image_callback, select_color=self.colors.selected, video=(type(media) == Video))
            self.preview_bar.controls.append(media.preview_container)
        self.switch_image(ids[-1])
        self.page.update()

    def switch_image(self, id: str):
        if self.selected_img is not None:
            if id == self.selected_img: return
            media = self.file_manager[self.selected_img]
            media.selected(False)
            if type(media) is Video:
                media.position = self.media_container.content.get_current_position()
        if id not in self.file_manager: # image was probably deleted
            self.selected_img = None
            self.media_container.content = None
        else:
            self.selected_img = id
            media = self.file_manager[id]
            media.selected(True)
            if type(media) is Image:
                self.media_container.content = ft.Image(media.get_path_preview(), fit=ft.ImageFit.CONTAIN)
            if type(media) is Video:
                self.media_container.content = VideoPlayer(media.get_path_preview(), media.aspect_ratio, colors=self.colors)
                # TODO: set player to current position
        self.page.update()

    def switch_image_callback(self, info: ft.ControlEvent):
        name = info.control.key
        self.switch_image(name)

    def export_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.path is None: return
        img = self.file_manager[self.selected_img]
        self.file_manager.export_image(img.id, file_results.path)
        if img.has_to_be_closed:
            self.close_image(img.id)

    def close_image(self, name):
        print(len(self.preview_bar.controls))
        self.preview_bar.controls = [c for c in self.preview_bar.controls if c.key != name]
        print(len(self.preview_bar.controls))
        self.selected_img = None
        del self.file_manager[name]
        self.switch_image(list(self.file_manager.keys())[0] if len(self.file_manager) >= 1 else '')

    def close_callback(self, info: ft.ControlEvent):
        if self.selected_img is None: return
        img = self.file_manager[self.selected_img]
        def save_callback():
            self.file_picker_export.save_file(file_name=img.get_orig_name())
            img.has_to_be_closed = True
        if img.saved:
            self.close_image(img.id)
        else:
            self.page.open(AlertSaveWindow(
                save_callback=save_callback,
                close_callback=lambda: self.close_image(img.id)
            ))

    def settings_callback(self, info: ft.ControlEvent):
        print('TODO: Settings')

    def build_page(self, page: ft.Page):
        self.page = page
        page.padding = 0
        page.spacing = 0
        page.bgcolor = self.colors.background
        # page.on_keyboard_event = lambda e: print(e)
        page.overlay.append(self.file_picker_open)
        page.overlay.append(self.file_picker_export)
        page.add(
            ft.Container(ft.Row([
                ft.Container(content=ft.IconButton(ft.icons.BLUR_ON, focus_color=self.colors.dark), width=50),
                ft.ElevatedButton("Open file", color=self.colors.text, on_click=lambda _: self.file_picker_open.pick_files(
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allowed_extensions = self.file_manager.IMAGE_FMTS + self.file_manager.VIDEO_FMTS,
                    allow_multiple=True), icon=ft.icons.FOLDER_OPEN),
                ft.ElevatedButton("Export file", color=self.colors.text, on_click=lambda _: self.file_picker_export.save_file(file_name=self.file_manager[self.selected_img].get_orig_name()), icon=ft.icons.SAVE_ALT),
                ft.ElevatedButton("Close file", color=self.colors.text, on_click=self.close_callback, icon=ft.icons.DELETE),
                ft.ElevatedButton("Blur all", color=self.colors.text, on_click=lambda _: self.blur_callback(), icon=ft.icons.PLAY_ARROW),
                ft.Row([], expand=True),
                ft.IconButton(on_click=self.settings_callback, icon=ft.icons.SETTINGS)
            ]), padding=10, bgcolor=self.colors.dark),
            ft.Row([
                ft.Container(self.preview_bar, bgcolor=self.colors.normal, padding=10, width=70),
                self.media_container,
                ft.Container(ft.Column([
                    ft.ExpansionTile(
                        title=ft.Text("Number Plates"),
                        leading=ft.Checkbox(),
                        shape=ft.StadiumBorder(),
                        controls=[ft.Text("options")]
                        ),
                    ft.ExpansionTile(
                        title=ft.Text("Faces"),
                        leading=ft.Checkbox(),
                        shape=ft.StadiumBorder(),
                        controls=[ft.Text("options")]
                        ),
                ], expand=True), bgcolor=self.colors.normal, width=300, expand=0.5, alignment=ft.alignment.top_left),
            ], expand=True),
        )
        names = self.file_manager.load_cached()
        self.load_images(names)

    def run(self):
        ft.app(target=self.build_page)