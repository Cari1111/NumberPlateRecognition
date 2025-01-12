import flet as ft
from .dataclasses import Image, Video, Media, FileVersion, FileVersionTemplate, ColorPalette, Version


class PreviewImage(ft.Stack):
    def __init__(self, key, path, callback, select_color, video: bool = False):
        self.select_color = select_color
        self.container = ft.Container(
            key=key,
            image_src=path,
            width=50,
            height=50,
            on_click=callback,
            image_fit=ft.ImageFit.COVER,
            border_radius=10,
        )
        self.triangle = ft.Container(ft.canvas.Canvas([
            ft.canvas.Path(
                [ft.canvas.Path.MoveTo(-3, 7), ft.canvas.Path.LineTo(-3, -7), ft.canvas.Path.LineTo(7, 0)],
                paint=ft.Paint(style=ft.PaintingStyle.FILL, color='#fcfcfc', stroke_cap=ft.StrokeCap.ROUND)),
            ft.canvas.Path(
                [ft.canvas.Path.MoveTo(-4, 8), ft.canvas.Path.LineTo(-4, -8), ft.canvas.Path.LineTo(8, 0), ft.canvas.Path.Close()],
                paint=ft.Paint(style=ft.PaintingStyle.STROKE, stroke_width=1, color='#3e3f40', stroke_cap=ft.StrokeCap.ROUND))]))
        super().__init__(
            key=key,
            controls=[self.container, self.triangle] if video else [self.container],
            alignment=ft.alignment.center
        )

    def toggle_selected(self, selected):
        self.container.border=(ft.border.all(3, color=self.select_color) if selected else None)

class AlertSaveWindow(ft.AlertDialog):
    def __init__(self, save_callback, close_callback):
        def save(e):
            self.page.close(self)
            save_callback()
        def close(e):
            self.page.close(self)
            close_callback()
        super().__init__(
            modal=True,
            title=ft.Text("File is not saved"),
            content=ft.Text("Do you want to save before closing the file?"),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.page.close(self)),
                ft.TextButton("Don't save", on_click=close),
                ft.TextButton("Save", on_click=save),
            ])

class VideoPlayer(ft.Video):
    def __init__(self, path, aspect_ratio, colors):
        super().__init__(
            playlist=[ft.VideoMedia(path)],
            fill_color=colors.background,
            aspect_ratio=aspect_ratio,
            volume=100,
            autoplay=False,
            filter_quality=ft.FilterQuality.HIGH,
            muted=False,
        )

class ModelTile(ft.ExpansionTile):
    def __init__(self, name, open_closed: dict, active: dict, colors: ColorPalette, active_callback, boundingBox_callback, blur_callback):
        def open_close_callback(info):
            open_closed[info.control.key] = info.data
        super().__init__(
            title=ft.Text(name, color=colors.text),
            initially_expanded = open_closed[name],
            maintain_state = True,
            key=name,
            leading=ft.Checkbox(on_change=active_callback, value=active[name], key=name),
            shape=ft.StadiumBorder(),
            expanded_cross_axis_alignment=ft.CrossAxisAlignment.START,
            controls_padding=5,
            on_change=open_close_callback,
            controls=[
                ft.OutlinedButton(
                    content=ft.Column([ft.Text("show bounding boxes", color=colors.text)]),
                    on_click=boundingBox_callback,
                    key=name),
                ft.Container(width=1, height=10),
                ft.OutlinedButton(
                    content=ft.Column([ft.Text("blur image", color=colors.text)]),
                    on_click=blur_callback,
                    key=name),
                ])