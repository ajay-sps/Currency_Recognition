from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from keras._tf_keras.keras.models import load_model


model = load_model('currency_recognition_model.h5')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=30,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    "dataset/training/",
    target_size=(64,64),
    batch_size=32,
    shuffle=True,
    seed=42,
    color_mode="rgb",
    class_mode='categorical')


# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64)) 
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 
    return img_array

# Function to predict the class
def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return prediction

# Example usage
# image_path = '50__295.jpg'  # Replace with your image path
# prediction = predict_image(model, image_path)
# print(f'Prediction: {prediction}')


class_indices = train_generator.class_indices
class_indices = {v: k for k, v in class_indices.items()}  # Invert the dictionary

# predicted_class_index = np.argmax(prediction, axis=1)[0]
# predicted_class_label = class_indices[predicted_class_index]
# print(f'Predicted class: {predicted_class_label}')

def predict_denomination(image_path):
    prediction = predict_image(model, image_path)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_indices[predicted_class_index]

    return predicted_class_label