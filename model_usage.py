import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

DOG_CLASSES = [
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
    model = load_model('C:/Users/kaiqu/dog-or-cat/modelos/dog_or_cat.keras')
    img = dog_cat_preprocessing(file, 300)

    # Exibe a imagem
    plt.imshow(img)
    plt.title("Imagem de Teste")
    plt.axis("off")
    plt.show()

    # Expande a dimensão da imagem para incluir o batch dimension (ficar shape: (1, 256, 256, 3))
    img_expanded = np.expand_dims(img, axis=0)

    pred_prob = model.predict(img_expanded)
    return pred_prob

def dog_breed_classifier(file):
    model = load_model('C:/Users/kaiqu/dog-or-cat/modelos/model_24_webscraped_classes.h5')

    img = breed_preprocessing(file, 300)
    img_expanded = np.expand_dims(img, axis=0)

    previsao = model.predict(img_expanded)

    indices_ordenados = np.argsort(previsao[0])

    top5_indices = indices_ordenados[-5:][::-1]

    print("Top 5 índices:\n", top5_indices)
    for i in top5_indices:
        print(f"Raça: {DOG_CLASSES[i]} - Probabilidade: {(previsao[0][i]) * 100:.2f}%")

def cat_breed_classifier(file):
    model = load_model('C:/Users/kaiqu/dog-or-cat/modelos/catbreed_model_v5.h5')

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
