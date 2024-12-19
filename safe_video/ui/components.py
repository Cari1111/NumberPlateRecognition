import flet as ft


class PreviewImage(ft.Container):
    def __init__(self, key, path, callback):
        super().__init__(
            key=key,
            image_src=path,
            width=50,
            height=50,
            on_click=callback,
            image_fit=ft.ImageFit.COVER,
            border_radius=10,
        )

    def toggle_selected(self, selected):
        self.border = (ft.border.all(3, ft.colors.BLUE_600) if selected else None)


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
