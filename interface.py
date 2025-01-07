import torch
from transformers import BertModel, AutoTokenizer
from torch import nn
import tkinter as tk
from PIL import Image, ImageTk
import time
from threading import Thread

# ========================== Cargar el Modelo y Tokenizador ==========================

class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 50),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout para evitar sobreajuste
            nn.Linear(50, 5)  # 5 clases de salida
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

model_path = "modelo-token/tokenization/bullying_tweets_model.pth"
tokenizer_path = "modelo-token/tokenization"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model = Bert_Classifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

print("Modelo y tokenizador cargados correctamente.")

# ========================== Definir las Categorías ==========================

LABELS = [
    "Bullying por Región",# Índice 0
    "Bullying por Edad",  # Índice 1
    "Bullying por Etnia", # Índice 2
    "Bullying de Género", # Índice 3
    "No Bullying",        # Índice 4
]

# ========================== Función para Analizar Tweets ==========================

def analyze_tweet():
    global start_time
    tweet_text = tweet_input.get("1.0", "end-1c").strip()
    
    if tweet_text:
        inputs = tokenizer(tweet_text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.softmax(outputs, dim=1)
            predicted_label = torch.argmax(prediction, dim=1).item()

        # Mostrar la categoría detectada
        response_label.config(text=f"Resultado: {LABELS[predicted_label]}")

        # Actualizar timestamp
        start_time = time.time()
        Thread(target=update_timestamp, daemon=True).start()
    else:
        response_label.config(text="Por favor, ingrese un tweet.")

# ========================== Función para Actualizar el Timestamp ==========================

def update_timestamp():
    global start_time
    while True:
        elapsed_time = int(time.time() - start_time)
        if elapsed_time < 60:
            timestamp_label.config(text=f"hace {elapsed_time} seg")
        else:
            timestamp_label.config(text=f"hace {elapsed_time // 60} min")
        time.sleep(1)

# ========================== Interfaz Gráfica con Tkinter ==========================

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
canvas = tk.Canvas(root, width=1200, height=700)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Crear un frame para los widgets
frame = tk.Frame(root, bg="#3A3A3A", bd=10)
frame.place(relx=0.5, rely=0.5, anchor="center")

# Input de texto del tweet
tweet_input = tk.Text(root, width=40, height=5, font=("Arial", 12), bg="#FFFFFF")
tweet_input.place(x=81, y=230, width=519, height=195)

# Cargar la imagen del botón
button_image_path = "design/button.png"
button_image = Image.open(button_image_path).resize((190, 80), Image.Resampling.LANCZOS)
button_photo = ImageTk.PhotoImage(button_image)

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
