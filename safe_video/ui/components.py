import flet as ft
from .color_picker.color_picker import ColorPicker
from .dataclasses import Image, Video, Media, FileVersion, FileVersionTemplate, ColorPalette, Version
import os
from safe_video.number_plate_recognition import Censor


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

class ColorPickerWindow(ft.AlertDialog):
    def __init__(self, initial_color, color_callback):
        self.color_picker = ColorPicker(initial_color) if initial_color else ColorPicker()
        def pick_color(info):
            color_callback(self.color_picker.color)
            self.page.close(self)
        super().__init__(
            title=ft.Text('Pick a color'),
            content=self.color_picker,
            actions=[
                ft.TextButton("Close", on_click=lambda _: self.page.close(self)),
                ft.TextButton("Pick Color", on_click=pick_color),
            ]
        )
class ModelTileTextStyle(ft.TextStyle):
    def __init__(self):
        super().__init__(weight=ft.FontWeight.W_500)
class ModelTileButton(ft.OutlinedButton):
    def __init__(self, colors: ColorPalette, text, on_click, key=None, disabled=False):
        super().__init__(
            content=ft.Text(text, color=colors.normal if not disabled else colors.background, style=ModelTileTextStyle()),
            on_click=on_click,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), side=ft.BorderSide(1, colors.text), padding=10),
            disabled=disabled,
            key=key),
class ModelTileIconButton(ft.IconButton):
    def __init__(self, colors: ColorPalette, icon, on_click, icon_color=None):
        super().__init__(
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), side=ft.BorderSide(1, colors.text)),
            icon=icon,
            height=45,
            width=45,
            on_click=on_click,
            icon_color=icon_color if icon_color else colors.text
        )

class CensorOptions(ft.Row):
    def __init__(self, page: ft.Page, colors: ColorPalette, update_func):
        self.main_page = page
        self.update_func = update_func
        str_options = ['blur', 'solid color', 'image']
        dropdown_options = [ft.dropdown.Option(s) for s in str_options]
        self.color = "#000000"
        self.option = str_options[0]
        self.file_path = None
        self.colors = colors
        self.file_picker = ft.FilePicker(on_result=self.get_path)
        self.main_page.overlay.append(self.file_picker)
        super().__init__(controls=[ft.Dropdown(
                value=self.option,
                options=dropdown_options,
                width=130,
                color=colors.text,
                border_color=colors.text,
                border_width=1,
                border_radius=10,
                focused_color=colors.text,
                height=45,
                padding=0,
                text_style=ModelTileTextStyle(),
                on_change=self.change_option)
            ]
        )

    def change_option(self, info: ft.ControlEvent):
        self.option = info.data
        match info.data:
            case 'solid color':
                self.add_extra_button(ModelTileIconButton(self.colors, icon=ft.icons.RECTANGLE, icon_color=self.color,
                    on_click=lambda _: self.main_page.open(ColorPickerWindow(self.color, self.pick_color))))
            case 'image':
                self.add_extra_button(ModelTileIconButton(self.colors, icon=ft.icons.FOLDER, icon_color=None if self.file_path else ft.colors.RED_500,
                    on_click=lambda _: self.file_picker.pick_files("Pick file to censor", allow_multiple=False)))
            case _:
                if len(self.controls) >= 1:
                    del self.controls[1]
                    self.update_func()

    def pick_color(self, color: str):
        self.color = color
        self.add_extra_button(ModelTileIconButton(self.colors, icon=ft.icons.RECTANGLE, icon_color=self.color,
            on_click=lambda _: self.main_page.open(ColorPickerWindow(self.color, self.pick_color))))

    def get_path(self, event: ft.FilePickerResultEvent):
        if event.files is None or len(event.files) == 0: return
        self.file_path = event.files[0].path
        self.add_extra_button(ModelTileIconButton(self.colors, icon=ft.icons.FOLDER, icon_color=None if self.file_path else ft.colors.RED_500,
            on_click=lambda _: self.file_picker.pick_files("Pick file to censor", allow_multiple=False)))

    def add_extra_button(self, button: ft.Control):
        if len(self.controls) <= 1:
            self.controls.append(button)
        else: self.controls[1] = button
        self.update_func()

    def get_option(self):
        match self.option:
            case 'solid color':
                return {'action': Censor.solid, 'color': self.color}
            case 'image':
                if (self.file_path is None) or (not os.path.exists(self.file_path)): return {'action': Censor.solid, 'color': '#000000'} # just sensor with a black box if no image is provided
                return {'action': Censor.overlay, 'overlayImage': self.file_path}
        return {'action': Censor.blur }


class ModelTile(ft.ExpansionTile):
    def __init__(self, name, open_closed: dict, censor_options: dict[str, CensorOptions], active: dict, colors: ColorPalette, active_callback, boundingBox_callback, blur_callback, edit_callback, delete_callback, blur_buttons_status):  
        def open_close_callback(info):
            open_closed[info.control.key] = info.data
        super().__init__(
            title=ft.Text(name, color=colors.text),
            initially_expanded=open_closed[name],
            maintain_state=True,
            key=name,
            leading=ft.Checkbox(on_change=active_callback, value=active[name], key=name),
            bgcolor=colors.light,
            shape=ft.BeveledRectangleBorder(0),
            controls_padding=0,
            on_change=open_close_callback,
            controls=[ft.Container(ft.Row([
                ft.Container(ft.Column([
                    censor_options[name],
                    ModelTileButton(colors, "show bounding boxes", on_click=boundingBox_callback, key=name, disabled=blur_buttons_status), 
                    ModelTileButton(colors, "censor", on_click=blur_callback, key=name),
                ])),
                ft.Column([], expand=True),
                ft.Container(ft.Column([
                    ft.IconButton(icon=ft.icons.EDIT, icon_color=colors.text, on_click=edit_callback, key=name),
                    ft.IconButton(icon=ft.icons.DELETE, icon_color=ft.colors.RED_500,
                                  on_click=delete_callback, key=name),
                ]), bgcolor=colors.normal, border_radius=10),
            ], vertical_alignment=ft.CrossAxisAlignment.START, spacing=0), padding=ft.Padding(left=40, top=5, right=15, bottom=10))])
        
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


class SettingsWindow(ft.AlertDialog):
    def __init__(self, colors: ColorPalette, load_callback, model_callback, del_callback, file_picker):
        self.colors = colors
        self.error_text = ft.Text('', color=ft.colors.RED_400, weight=ft.FontWeight.BOLD)
        self.callback = load_callback
        self.model_callback = model_callback
        self.del_callback = del_callback
        self.models = []
        self.model_list = ft.Column()
        self.expanded_models = {}
        super().__init__(
            modal=False,
            title=ft.Text("Options", size=24, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
            content=ft.Container(
                content=ft.Column([
                    ft.Divider(),
                    ft.TextButton(
                        'Add new model',
                        icon=ft.icons.ADD,
                        on_click=lambda _: file_picker.pick_files(
                            file_type=ft.FilePickerFileType.CUSTOM,
                            allowed_extensions=['pt'],
                            allow_multiple=True
                        ),
                    ),
                    self.error_text,
                    ft.Text("Models:", size=18, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                    ft.Container(
                        content=ft.ListView([self.model_list], expand=True),
                        expand=True,
                        height=300
                    )
                ], width=500, spacing=10, alignment=ft.MainAxisAlignment.CENTER),
                alignment=ft.alignment.center
            ),
            actions=[
                ft.TextButton("Reload", on_click=lambda _: self.update_category_list()),
                ft.TextButton("Close", on_click=lambda _: self.page.close(self)),
            ]
        )
        self.update_category_list(True)

    def toggle_expand(self, model_name):
        self.expanded_models[model_name] = not self.expanded_models.get(model_name, True)
        self.update_category_list()

    def delete_model(self, model_name):
        self.del_callback(model_name)
        self.update_category_list()

    def add_model(self, file_results: ft.FilePickerResultEvent):
        self.callback(file_results.path)
        self.update_category_list()

    def update_category_list(self, init=False):
        self.model_list.controls.clear()
        self.models = self.model_callback()
        for model_name, elements in self.models:
            is_expanded = self.expanded_models.get(model_name, True)
            model_card = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text(model_name, weight=ft.FontWeight.BOLD, expand=True),
                        ft.IconButton(
                            icon=ft.icons.ARROW_DROP_DOWN if is_expanded else ft.icons.ARROW_RIGHT,
                            on_click=lambda e, name=model_name: self.toggle_expand(name)
                        ),
                        ft.IconButton(icon=ft.icons.DELETE, on_click=lambda e, name=model_name: self.delete_model(name))
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Container(
                        content=ft.Row([
                            ft.Container(
                                content=ft.Text(f"{elem}"),
                                padding=ft.padding.all(5),
                                bgcolor=self.colors.background,
                                border_radius=5
                            ) for elem in elements
                        ], spacing=5, wrap=True),
                        visible=is_expanded,
                        padding=ft.padding.only(left=20)
                    )
                ], spacing=10),
                padding=ft.padding.all(10),
                border=ft.border.all(1, ft.colors.GREY_400),
                border_radius=10,
                width=450
            )
            self.model_list.controls.append(model_card)
        if not init: self.update()