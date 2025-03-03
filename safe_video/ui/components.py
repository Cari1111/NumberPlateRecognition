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
                [ft.canvas.Path.MoveTo(-4, 8), ft.canvas.Path.LineTo(-4, -8),
                 ft.canvas.Path.LineTo(8, 0), ft.canvas.Path.Close()],
                paint=ft.Paint(style=ft.PaintingStyle.STROKE, stroke_width=1, color='#3e3f40', stroke_cap=ft.StrokeCap.ROUND))]))
        super().__init__(
            key=key,
            controls=[self.container, self.triangle] if video else [self.container],
            alignment=ft.alignment.center
        )

    def toggle_selected(self, selected):
        self.container.border = (ft.border.all(3, color=self.select_color) if selected else None)


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
    def __init__(self, name, open_closed: dict, active: dict, colors: ColorPalette, active_callback, boundingBox_callback, blur_callback, edit_callback, delete_callback, blur_buttons_status):
        def open_close_callback(info):
            open_closed[info.control.key] = info.data
        super().__init__(
            title=ft.Text(name, color=colors.text),
            initially_expanded=open_closed[name],
            maintain_state=True,
            key=name,
            leading=ft.Checkbox(on_change=active_callback, value=active[name], key=name),
            shape=ft.StadiumBorder(),
            expanded_cross_axis_alignment=ft.CrossAxisAlignment.START,
            controls_padding=5,
            on_change=open_close_callback,
            controls=[ft.Row([
                ft.Column([], width=30),
                ft.Container(ft.Column([
                    ft.OutlinedButton(
                        content=ft.Column([ft.Text("show bounding boxes", color=colors.text if not blur_buttons_status else colors.background )]),
                        on_click=boundingBox_callback,
                        disabled=blur_buttons_status,
                        key=name),
                    ft.OutlinedButton(
                        content=ft.Column([ft.Text("blur image", color=colors.text)]),
                        on_click=blur_callback,
                        key=name),
                ])),
                ft.Container(ft.Column([
                    ft.IconButton(icon=ft.icons.EDIT, icon_color=colors.text, on_click=edit_callback, key=name),
                    ft.IconButton(icon=ft.icons.DELETE, icon_color=ft.colors.RED_500,
                                  on_click=delete_callback, key=name)
                ]), bgcolor=colors.background, border_radius=10),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER)])


class ClassDropdown(ft.Dropdown):
    def __init__(self, cls_options: list[str], colors: ColorPalette, cls: str = ''):
        super().__init__(
            value=cls,
            options=[ft.dropdown.Option(cls_option) for cls_option in cls_options],
            width=150,
            color=colors.text,
            border_color=colors.background)


class AddClassLayerRow(ft.Container):
    def __init__(self, layer_num, cls_options: list[str], colors: ColorPalette, classes: list[str] | None = None):
        self.cls_options = cls_options
        self.colors = colors
        dropdowns = [ClassDropdown(cls_options, colors), ClassDropdown(cls_options, colors)]
        if classes:
            dropdowns = [ClassDropdown(cls_options, colors, cls) for cls in classes]
        self.cls_row = ft.Row(
            dropdowns + [
                ft.IconButton(ft.icons.ADD, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(
                    radius=10), padding=15), on_click=self.add_class_option),
                ft.IconButton(ft.icons.REMOVE, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(
                    radius=10), padding=15), on_click=self.remove_class_option)
            ], wrap=True, expand=True)
        super().__init__(
            ft.Column([ft.Text(f'Layer {layer_num}', weight=ft.FontWeight.BOLD, color=colors.text), self.cls_row]),
            bgcolor=colors.normal,
            border=ft.border.all(3, color=colors.background),
            border_radius=10,
            padding=10)

    def add_class_option(self, info):
        self.cls_row.controls.insert(-2, ClassDropdown(self.cls_options, self.colors))
        self.update()

    def remove_class_option(self, info):
        if len(self.cls_row.controls) <= 3: return
        del self.cls_row.controls[-3]
        self.update()

    def get_values(self):
        vals = []
        for dropdown in self.cls_row.controls[:-2]:
            if dropdown.value != '':
                vals.append(dropdown.value)
        return vals


class AddClassWindow(ft.AlertDialog):
    def __init__(self, class_options: list[str], add_class_callback, colors: ColorPalette, from_existing: tuple[str, list[list[str]]] = None):
        name = from_existing[0] if from_existing else ''
        self.num_layers = len(from_existing[1]) if from_existing else 2
        layers = [AddClassLayerRow(i, class_options, colors) for i in range(1, self.num_layers + 1)]
        if from_existing:
            layers = [AddClassLayerRow(i + 1, class_options, colors, cls) for i, cls in enumerate(from_existing[1])]
        self.class_options = class_options
        self.colors = colors
        self.error_text = ft.Text('', color=ft.colors.RED_400, weight=ft.FontWeight.BOLD)

        def add_class(e):
            name = self.content.controls[0].value
            if name == '':
                self.error_text.value = 'A name has to be chosen!'
                self.update()
                return
            classes = []
            for row in self.content.controls[1:-2]:
                vals = row.get_values()
                if len(vals) == 0: continue
                classes.append(vals)
            if len(classes) == 0:
                self.error_text.value = 'Layers can not all be empty!'
                self.update()
                return
            self.page.close(self)
            add_class_callback(name, classes)
        super().__init__(
            modal=True,
            title=ft.Text("Define the class"),
            content=ft.ListView(
                [ft.TextField(label="Name", value=name, color=colors.text,
                              bgcolor=colors.normal, border_color=colors.text)]
                + layers
                + [ft.Row([ft.TextButton(
                    'Add new layer',
                    icon=ft.icons.ADD,
                    on_click=self.add_layer,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), padding=20)),
                    ft.TextButton(
                    'Remove layer',
                    icon=ft.icons.REMOVE,
                    on_click=self.remove_layer,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), padding=20))]),
                   self.error_text
                   ], width=1000, spacing=10),
            actions=[
                ft.TextButton("Close", on_click=lambda _: self.page.close(self)),
                ft.TextButton("Edit class" if from_existing else "Add Class", on_click=add_class),
            ])

    def add_layer(self, info):
        self.num_layers += 1
        self.content.controls.insert(-2, AddClassLayerRow(self.num_layers, self.class_options, self.colors))
        self.update()

    def remove_layer(self, info):
        if self.num_layers <= 1: return
        self.num_layers -= 1
        del self.content.controls[-3]
        self.update()
