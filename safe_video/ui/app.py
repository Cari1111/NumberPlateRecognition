import flet as ft
from safe_video.number_plate_recognition import NumberPlateRecognition
from .dataclasses import Video, Image, ColorPalette
from .components import PreviewImage, AlertSaveWindow
from .helper_classes import FileManger

CACHE_PATH = "safe_video/upload_cache/"

DarkColors = ColorPalette(
    normal = "#1a1e26",
    light = "#232833",
    dark = "#101217",
)

class UI_App:
    def __init__(self):
        self.file_manager = FileManger(CACHE_PATH)
        self.page: ft.Page = None
        self.colors: ColorPalette = DarkColors
        self.image_container = ft.Container(expand=True, image_fit=ft.ImageFit.CONTAIN, margin=10)
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
        images = []
        for file in file_results.files:
            name = self.file_manager.upload_image(old_path=file.path, filename=file.name)
            images.append(name)
        self.load_images(images)

    def load_images(self, names: list[str]):
        if len(names) == 0: return
        for name in names:
            img = self.file_manager[name]
            img.preview_container = PreviewImage(img.id, img.get_path_preview(), self.switch_image_callback)
            self.preview_bar.controls.append(ft.Row([ft.Checkbox(), img.preview_container]))
        self.switch_image(names[-1])
        self.page.update()

    def switch_image(self, id: str):
        if self.selected_img is not None:
            if id == self.selected_img: return
            self.file_manager[self.selected_img].selected(False)
        if id not in self.file_manager: # image was probably deleted
            self.selected_img = None
            self.image_container.content = None
        else:
            self.selected_img = id
            img = self.file_manager[id]
            img.selected(True)
            self.image_container.content = ft.Image(img.get_path_preview(), fit=ft.ImageFit.CONTAIN)
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
        self.preview_bar.controls = [c for c in self.preview_bar.controls if c.key != name]
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
        page.bgcolor = self.colors.light
        page.on_keyboard_event = lambda e: print(e)
        page.overlay.append(self.file_picker_open)
        page.overlay.append(self.file_picker_export)
        page.add(
            ft.Container(ft.Row([
                ft.Container(content=ft.IconButton(ft.icons.BLUR_ON, focus_color=self.colors.dark), width=50),
                ft.ElevatedButton("Open Image", on_click=lambda _: self.file_picker_open.pick_files(file_type=ft.FilePickerFileType.IMAGE, allow_multiple=True), icon=ft.icons.FOLDER_OPEN),
                ft.ElevatedButton("Export file", on_click=lambda _: self.file_picker_export.save_file(file_name=self.file_manager[self.selected_img].get_orig_name()), icon=ft.icons.SAVE_ALT),
                ft.ElevatedButton("Close file", on_click=self.close_callback, icon=ft.icons.DELETE),
                ft.ElevatedButton("Blur all", on_click=lambda _: self.blur_callback(), icon=ft.icons.PLAY_ARROW),
                ft.Row([], expand=True),
                ft.IconButton(on_click=self.settings_callback, icon=ft.icons.SETTINGS)
            ]), padding=10, bgcolor=self.colors.dark),
            ft.Row([
                ft.Container(self.preview_bar, bgcolor=self.colors.normal, padding=10, width=120),
                self.image_container,
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