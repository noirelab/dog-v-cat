import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

DOG_CLASSES = [
    "Chihuahua",
    "Maltese_dog",
    "Pekinese",
    "Shih-Tzu",
    "papillon",
    "Afghan_hound",
    "beagle",
    "bloodhound",
    "Yorkshire_terrier",
    "miniature_schnauzer",
    "golden_retriever",
    "Labrador_retriever",
    "German_shepherd",
    "French_bulldog",
    "Siberian_husky",
    "standard_poodle",
    "boxer",
    "cocker_spaniel",
    "Saint_Bernard",
    "pug",
    "Rottweiler",
    "malinois",
    "Border_collie",
    "miniature_pinscher"
]

CAT_CLASSES = [
    'Abyssinian cat',
    'American Shorthair cat',
    'Bengal cat',
    'Birman cat',
    'British Shorthair cat',
    'Burmese cat',
    'Calico',
    'Chartreux cat',
    'Cornish Rex cat',
    'Devon Rex cat',
    'Domestic Medium Hair',
    'Egyptian Mau cat',
    'Japanese Bobtail cat',
    'Maine Coon cat',
    'Munchkin cat',
    'Ocicat cat',
    'Persian cat',
    'Russian Blue cat',
    'Scottish Fold cat',
    'Siamese cat',
    'Sphynx cat',
    'Turkish Angora cat',
    'Turkish Van cat',
    'Tuxedo'
]

def dog_cat_preprocessing(image_path, size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    image = image.astype('float32') / 255.0

    return image

def breed_preprocessing(file, size):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img

def dog_or_cat_classifier(file):
    model = load_model('C:/Users/kaiqu/dog-v-cat/modelos/dog_or_cat.keras')
    img = dog_cat_preprocessing(file, 300)

    plt.imshow(img)
    plt.title("Imagem de Teste")
    plt.axis("off")
    plt.show()

    # Tem que expandir pra pra poder caber no modelo
    img_expanded = np.expand_dims(img, axis=0)

    pred_prob = model.predict(img_expanded)
    return pred_prob

def dog_breed_classifier(file):
    model = load_model('C:/Users/kaiqu/dog-v-cat/modelos/model_24_webscraped_classes.h5')

    img = breed_preprocessing(file, 300)
    img_expanded = np.expand_dims(img, axis=0)

    previsao = model.predict(img_expanded)

    indices_ordenados = np.argsort(previsao[0])

    top5_indices = indices_ordenados[-5:][::-1]

    print("Top 5 índices:\n", top5_indices)
    for i in top5_indices:
        print(f"Raça: {DOG_CLASSES[i]} - Probabilidade: {(previsao[0][i]) * 100:.2f}%")

def cat_breed_classifier(file):
    model = load_model('C:/Users/kaiqu/dog-v-cat/modelos/catbreed_model_v5.h5')

    img = breed_preprocessing(file, 224)
    img_expanded = np.expand_dims(img, axis=0)

    previsao = model.predict(img_expanded)

    indices_ordenados = np.argsort(previsao[0])

    top5_indices = indices_ordenados[-5:][::-1]

    print("Top 5 índices:\n", top5_indices)
    for i in top5_indices:
        print(f"Raça: {CAT_CLASSES[i]} - Probabilidade: {(previsao[0][i]) * 100:.2f}%")

def dog_cat_breed_classifier(image):
    pred_prob = dog_or_cat_classifier(image)

    print(f'Índice (0 = gato, 1 = cachorro): {pred_prob[0][0]:.4f}')

    if pred_prob[0][0] >= 0.5:
        print("Rótulo previsto: Cachorro\n")
        dog_breed_classifier(image)
    else:
        print("Rótulo previsto: Gato\n")
        cat_breed_classifier(image)

""" def dog_cat_breed_classifier(image_path):
    pred_prob = dog_or_cat_classifier(image_path)
    label = "Cachorro" if pred_prob[0][0] >= 0.5 else "Gato"
    detalhes = []
    if label == "Cachorro":
        detalhes = dog_breed_classifier(image_path)  # faça essa função devolver lista em vez de print
    else:
        detalhes = cat_breed_classifier(image_path)
    return label, detalhes
 """
