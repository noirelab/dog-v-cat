import flet as ft
import base64
from gui_functions import dog_cat_breed_classifier

def main(page: ft.Page):
    page.title = "Projeto Final LAMIA - dog-v-cat"
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER
    page.update()

    # Escolhe o arquivo que vai fazer a predição
    picker = ft.FilePicker(on_result=lambda e: on_file_picked(e, img, result_txt))
    page.overlay.append(picker)

    # Apenas para consertar o bug do flet que não mostra a imagem
    img = ft.Image(src="", width=300, height=300, visible=False)
    result_txt = ft.Text("", size=16)

    def on_file_picked(e: ft.FilePickerResultEvent, img_ctrl, txt_ctrl):
        if not e.files:
            return
        file = e.files[0]

        # Quando escolher uma nova image, apaga o texto anterior
        txt_ctrl.value = ""
        txt_ctrl.update()

        with open(file.path, "rb") as f:
            data = f.read()

        img_ctrl.src_base64 = base64.b64encode(data).decode()
        img_ctrl.visible = True
        img_ctrl.update()

        label, details = dog_cat_breed_classifier(file.path)
        lines = [f"Predição: {label}"] + [f"{b}: {p:.2f}%" for b, p in details]
        txt_ctrl.value = "\n".join(lines)
        txt_ctrl.update()


    # Botão que passa em lambda pra função
    btn = ft.ElevatedButton(
        "Selecionar Imagem",
        on_click=lambda _: picker.pick_files(allow_multiple=False)
    )

    page.add(
        ft.Column(
            [
                ft.Text("dog-v-cat", size=30, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                ft.Divider(),
                btn,
                ft.Divider(),
                ft.Container(img, padding=10, bgcolor=ft.colors.GREY_200),
                ft.Divider(),
                result_txt
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
