from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

CLASSES = [
    "n02085620-Chihuahua",
    "n02085936-Maltese_dog",
    "n02086079-Pekinese",
    "n02086240-Shih-Tzu",
    "n02086910-papillon",
    "n02088094-Afghan_hound",
    "n02088364-beagle",
    "n02088466-bloodhound",
    "n02094433-Yorkshire_terrier",
    "n02097047-miniature_schnauzer",
    "n02099601-golden_retriever",
    "n02099712-Labrador_retriever",
    "n02106662-German_shepherd",
    "n02108915-French_bulldog",
    "n02110185-Siberian_husky",
    "n02113799-standard_poodle",
    "n02108089-boxer",
    "n02102318-cocker_spaniel",
    "n02109525-Saint_Bernard",
    "n02110958-pug",
    "n02106550-Rottweiler",
    "n02105162-malinois",
    "n02106166-Border_collie",
    "n02107312-miniature_pinscher"
]

def load_preprocess_image(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def dog_breed_classifier(file):
    model = load_model('C:/Users/kaiqu/dog-or-cat/modelos/model_24_webscraped_classes.h5')

    img = load_preprocess_image(file)

    img_expanded = np.expand_dims(img, axis=0)

    # Realiza a predição
    previsao = model.predict(img_expanded)
    print("Predição (vetor de probabilidades):")
    # print(previsao)

    # # Ordena os índices de acordo com as probabilidades (do menor para o maior)
    indices_ordenados = np.argsort(previsao[0])
    # # Seleciona os 5 índices com maiores probabilidades e inverte a ordem (do maior para o menor)
    top5_indices = indices_ordenados[-5:][::-1]
    print("Top 5 índices:", top5_indices)

    # print("Top 5 raças mais prováveis:")
    for i in top5_indices:
        print(f"Raça: {CLASSES[i]} - Probabilidade: {(previsao[0][i]) * 100:.2f}%")
