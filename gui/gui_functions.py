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
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype("float32") / 255.0
    return img

def breed_preprocessing(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img

def dog_or_cat_classifier(path):
    model = load_model("C:/Users/kaiqu/dog-v-cat/modelos/dog_or_cat.keras")
    img = dog_cat_preprocessing(path, 300)
    prob = model.predict(img[None, ...])[0][0]

    return prob

def top5(preds, class_names):
    idxs = np.argsort(preds)[-5:][::-1]

    return [(class_names[i], float(preds[i] * 100)) for i in idxs]

def dog_breed_classifier(path):
    model = load_model("C:/Users/kaiqu/dog-v-cat/modelos/model_24_webscraped_classes.h5")
    img = breed_preprocessing(path, 300)
    preds = model.predict(img[None, ...])[0]

    return top5(preds, DOG_CLASSES)

def cat_breed_classifier(path):
    model = load_model("C:/Users/kaiqu/dog-v-cat/modelos/catbreed_model_v5.h5")
    img = breed_preprocessing(path, 224)
    preds = model.predict(img[None, ...])[0]

    return top5(preds, CAT_CLASSES)

def dog_cat_breed_classifier(path):
    p = dog_or_cat_classifier(path)
    label = "Cachorro" if p >= 0.5 else "Gato"
    details = dog_breed_classifier(path) if p >= 0.5 else cat_breed_classifier(path)

    return label, details
