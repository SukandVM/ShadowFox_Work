import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
model = tf.keras.models.load_model("SVMcifar10.h5")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict(img):
    img = img.resize((32, 32)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    prediction = model.predict(img_array)[0]
    return {class_names[i]: float(prediction[i]) for i in range(10)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier",
    description="Upload an image (32x32 or larger). The model predicts the class among 10 categories."
).launch()
