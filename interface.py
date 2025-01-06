import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import time
from threading import Thread


def update_timestamp():
    global start_time
    while True:
        elapsed_time = int(time.time() - start_time)
        if elapsed_time < 60:
            timestamp_label.config(text=f"hace {elapsed_time} seg")
        else:
            timestamp_label.config(text=f"hace {elapsed_time // 60} min")
        time.sleep(1)


def analyze_tweet():
    global start_time
    tweet_text = tweet_input.get("1.0", "end-1c")
    if tweet_text.strip():
        response_label.config(text="El tweet contiene bullying...")
        start_time = time.time()
        Thread(target=update_timestamp, daemon=True).start()
    else:
        response_label.config(text="Por favor, ingrese un tweet.")


# Crear la ventana principal
root = tk.Tk()
root.title("Detector de Bullying en Tweets")
root.geometry("1200x700")
root.resizable(False, False)

# Cargar la imagen de fondo
bg_image_path = "design/interface-w-b.png"  
bg_image = Image.open(bg_image_path).resize((1200, 700), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Crear un canvas para el fondo
canvas = tk.Canvas(root, width=1000, height=800)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Crear un frame para los widgets
frame = tk.Frame(root, bg="#3A3A3A", bd=10)
frame.place(relx=0.5, rely=0.5, anchor="center")


# Cargar la imagen del botón
button_image_path = "design/button.png"  # Cambia esta ruta
button_image = Image.open(button_image_path).resize((190, 80), Image.Resampling.LANCZOS)
button_photo = ImageTk.PhotoImage(button_image)

analyze_button = tk.Button(
    root, 
    image=button_photo, 
    command=analyze_tweet, 
    borderwidth=0, 
    bg="#3A3A3A", 
    activebackground="#3A3A3A",  # Evitar cambio de color al hacer clic
    highlightthickness=0         # Quitar el borde del enfoque
)

# Input de texto del tweet
tweet_input = tk.Text(root, width=40, height=5, font=("Arial", 12), bg="#FFFFFF")
tweet_input.place(x=81, y=230, width=519, height=195)

# Botón para analizar el tweet
analyze_button = tk.Button(
    root, image=button_photo, command=analyze_tweet, borderwidth=0, bg="#3A3A3A"
)
analyze_button.place(x=410, y=550, width=188, height=60)

# Etiqueta para respuesta
response_label = tk.Label(root, text="", font=("Arial", 13), fg="#FFFFFF", bg="#747474")
response_label.place(x=718, y=230, width=393, height=320)

# Etiqueta para el timestamp
timestamp_label = tk.Label(root, text="", font=("Arial", 13), fg="#AAAAAA", bg="#747474")
timestamp_label.place(x=718, y=549, width=393, height=30)


start_time = 0

# Ejecutar la aplicación
root.mainloop()
