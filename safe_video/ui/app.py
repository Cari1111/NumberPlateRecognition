import flet as ft
from safe_video.number_plate_recognition import ObjectDetection
from .dataclasses import Video, Image, ColorPalette, Version
from .components import PreviewImage, AlertSaveWindow, VideoPlayer, ModelTile
from .helper_classes import FileManger, ModelManager
from flet.matplotlib_chart import MatplotlibChart
import base64

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
        self.model_manager = ModelManager(self.show_bounding_boxes)
        self.page: ft.Page = None
        self.media_container = ft.Container(expand=True, image_fit=ft.ImageFit.CONTAIN, margin=10)
        self.preview_bar = ft.ListView([], expand=True, spacing=10)
        self.selected_media: str = None
        self.file_picker_open = ft.FilePicker(on_result=self.upload_callback)
        self.file_picker_export = ft.FilePicker(on_result=self.export_callback)
        self.tiles_open_closed = {cls: False for cls in self.model_manager.cls}
        self.tiles: ft.ListView = ft.ListView([], expand=True)
        self.show_censored = True

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
            media.preview_container = PreviewImage(media.id, media.get_path(Version.ICON), self.switch_image_callback, select_color=self.colors.selected, video=(type(media) == Video))
            self.preview_bar.controls.append(media.preview_container)
        self.switch_image(ids[-1])
        self.update()

    def switch_image(self, id: str):
        if self.selected_media is not None:
            if id == self.selected_media: return
            media = self.file_manager[self.selected_media]
            media.selected(False)
            if type(media) is Video:
                media.position = self.media_container.content.get_current_position()
        if id not in self.file_manager: # image was probably deleted
            self.selected_media = None
            self.media_container.content = None
        else:
            self.selected_media = id
            media = self.file_manager[id]
            media.selected(True)
            if type(media) is Image:
                self.media_container.content = ft.Image(media.get_path_preview(self.show_censored), fit=ft.ImageFit.CONTAIN)
            if type(media) is Video:
                self.media_container.content = VideoPlayer(media.get_path_preview(self.show_censored), media.aspect_ratio, colors=self.colors)
                # TODO: set player to current position
        self.update()

    def switch_image_callback(self, info: ft.ControlEvent):
        name = info.control.key
        self.switch_image(name)

    def export_callback(self, file_results: ft.FilePickerResultEvent):
        if file_results.path is None: return
        img = self.file_manager[self.selected_media]
        self.file_manager.export_image(img.id, file_results.path)
        if img.has_to_be_closed:
            self.close_image(img.id)

    def close_image(self, name):
        self.preview_bar.controls = [c for c in self.preview_bar.controls if c.key != name]
        self.selected_media = None
        del self.file_manager[name]
        self.switch_image(list(self.file_manager.keys())[0] if len(self.file_manager) >= 1 else '')

    def close_callback(self, info: ft.ControlEvent):
        if self.selected_media is None: return
        img = self.file_manager[self.selected_media]
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

    def show_bounding_boxes(self, model_id):
        fig = self.model_manager.get_bounding_box_fig(model_id, self.file_manager[self.selected_media])
        self.media_container.content = MatplotlibChart(fig, expand=True)
        self.update()

    def blur_img(self, img: Image, cls_ids: list[str]):
        censored_img = self.model_manager.get_blurred_as_list(cls_ids, img)
        self.file_manager.create_blurred_imgs(img.id, censored_img)

    def blur_current_img_callback(self, cls_id):
        self.blur_img(self.file_manager[self.selected_media], [cls_id])
        self.update_media_container_with_img()

    def blur_all_callback(self):
        for img in self.file_manager.values():
            if type(img) is not Image: continue
            self.blur_img(img, [cls_id for cls_id in self.model_manager.cls if self.model_manager.active[cls_id]])
        self.update_media_container_with_img()

    def toggle_blur_orig(self, info):
        self.show_censored = not self.show_censored
        self.update_media_container_with_img()

    def add_model_callback(self, info):
        print('add model')

    def settings_callback(self, info: ft.ControlEvent):
        print('TODO: Settings')

    def update(self):
        self.tiles.controls = [
            ModelTile(c, self.tiles_open_closed, self.model_manager.active, self.colors,
            active_callback=lambda info: self.model_manager.toggle_active(info.control.key),
            boundingBox_callback=lambda info: self.show_bounding_boxes(info.control.key),
            blur_callback=lambda info: self.blur_current_img_callback(info.control.key),
            ) for c in self.model_manager.cls]
        self.page.update()

    def update_media_container_with_img(self):
        with open(self.file_manager[self.selected_media].get_path_preview(self.show_censored), "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        self.media_container.content = ft.Image(src_base64=encoded_string, fit=ft.ImageFit.CONTAIN)
        self.media_container.update()
        self.update()

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
                ft.ElevatedButton("Export file", color=self.colors.text, on_click=lambda _: self.file_picker_export.save_file(file_name=self.file_manager[self.selected_media].get_orig_name()), icon=ft.icons.SAVE_ALT),
                ft.ElevatedButton("Close file", color=self.colors.text, on_click=self.close_callback, icon=ft.icons.DELETE),
                ft.VerticalDivider(width=9, thickness=1, color=self.colors.background),
                ft.ElevatedButton("Blur all", color=self.colors.text, on_click=lambda _: self.blur_all_callback(), icon=ft.icons.PLAY_ARROW),
                ft.Switch(active_color=self.colors.text, value=True, on_change=self.toggle_blur_orig),
                ft.Row([], expand=True),
                ft.IconButton(on_click=self.settings_callback, icon=ft.icons.SETTINGS)
            ]), padding=10, height=60, bgcolor=self.colors.dark),
            ft.Row([
                ft.Container(self.preview_bar, bgcolor=self.colors.normal, padding=10, width=70),
                self.media_container,
                ft.Container(ft.Column([
                    self.tiles,
                    ft.Container(ft.Row([ft.TextButton(
                        'Add new class',
                        icon=ft.icons.ADD,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), padding=20),
                        expand=True,
                        on_click=self.add_model_callback)]), padding=10)
                ], expand=True), bgcolor=self.colors.normal, width=300, expand=0.5, alignment=ft.alignment.top_left),
            ], expand=True),
        )
        names = self.file_manager.load_cached()
        self.load_images(names)
        self.update()

    def run(self):
        ft.app(target=self.build_page)