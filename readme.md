# dog-v-cat

This project uses machine learning to classify images of dogs and cats. The model was trained using the **MobileNetV2** and **EfficientNetV2B3** architectures and implemented with **TensorFlow**.
The initial dataset was very small, so a web-scraping script was used to increase the dog-breed dataset by approximately 416%.

### Dataset link
https://drive.google.com/file/d/1MDQkbVx87fWwb_wQ9K23tvzA2fbaxapl/view?usp=drivesdk

## Getting Started

```bash
cd gui
pip install -r requirements.txt
python gui.py
```

### Example in the interface

- ![Confusion matrix](images/imagens%20para%20o%20relatório/gui.png)

## Main Features

1. **Data Pre-processing**
   - Resize images to 300×300 px or 224×224 px (depending on the model).
   - Normalize pixel values.

2. **Dataset Splitting**
   - Split data into training and testing sets.

3. **Classification Model**
   - Use **MobileNetV2** as the base.
   - Add dense layers for fine-tuning and binary classification.

4. **Model Training**
   - Use callbacks like **EarlyStopping**, **ReduceLROnPlateau**, and **ModelCheckpoint** to optimize training.

5. **Evaluation & Visualization**
   - Generate accuracy and loss plots.
   - Display a confusion matrix and compute metrics such as **Recall**.

6. **Real-Image Prediction**
   - Classify new images using the trained model.

## Code Structure

- **Library Imports**: TensorFlow, OpenCV, Matplotlib, etc.
- **Data Loading**: Read images and map labels to integer values.
- **Training**: Train the model on the training set.
- **Evaluation**: Evaluate on the test set and show performance metrics.
- **Real-World Testing**: Classify user-provided external images.

## dog-v-cat Model Results

- **Training Accuracy**: 99.58%
- **Validation Accuracy**: 98.96%
- **Recall**: 99.2%

## Notes

- Make sure your dataset folders are organized correctly before running the code.
- The trained model will be saved as `dog_or_cat.keras`.

## Notebook Summaries

- **main.ipynb**
    - **Goal**: Classify images as “dog” or “cat.”
    - **Pipeline**:
        1. **Pre-processing**: Resize to 300×300 px and normalize pixels.
        2. **Model**: Based on **MobileNetV2** with added dense layers for binary classification.
        3. **Training**: Callbacks (**EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint**); saves model to `dog_or_cat.keras`.
        4. **Evaluation**: Show accuracy, confusion matrix, and **Recall**.
        5. **Prediction**: Classify new images as “dog” or “cat,” then use breed models (`dogbreed` or `catbreed`).
        ![Confusion matrix](images/imagens%20para%20o%20relatório/dog_cat_conf_matrix.png)

- **dogbreed_v3_reduced_enhanced.ipynb**
    - **Goal**: Classify images of 24 dog breeds.
    - **Pipeline**:
        1. **Pre-processing**: Create a DataFrame, split into train (80%) and test (20%).
        2. **Model**: Based on **EfficientNetV2B3**, with dense layers; Adam optimizer and `categorical_crossentropy` loss.
        3. **Training**: Callbacks (**EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint**); saves model to `model_24_webscraped_classes.h5`.
        4. **Results**: Show training accuracy & loss, evaluate on test set.
        ![Confusion matrix](images/imagens%20para%20o%20relatório/dog_conf_matrix.png)

- **catbreed_v5.ipynb**
    - **Goal**: Classify images of 24 cat breeds.
    - **Pipeline**:
        1. **Pre-processing**: Create a DataFrame, split into train (80%) and test (20%), one-hot encode classes.
        2. **Model**: Based on **EfficientNetV2B3**, with dense layers; Adam optimizer and `categorical_crossentropy` loss.
        3. **Training**: Callbacks (**EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint**); saves model to `catbreed_model_v5.h5`.
        4. **Results**: Show training accuracy & loss, evaluate on test set.
        ![Confusion matrix](images/imagens%20para%20o%20relatório/cat_conf_matrix.png)

## Usage Example in main.ipynb

```python
import model_usage as mu

# Classify an image
mu.dog_cat_breed_classifier('image.jpg')
```

This project demonstrates how to use convolutional neural networks to efficiently solve image-classification problems.
