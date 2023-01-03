import gradio as gr
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

img_height,img_width=180,180



model_flower = keras.models.load_model('model_flower.h5')







def predict_image(img):
  img_2d=img.reshape(-1,180,180,3)
  prediction=model_flower.predict(img_2d)[0]
  return {class_names[i]: float(prediction[i]) for i in range(5)}

image = gr.inputs.Image(shape=(180,180))
label = gr.outputs.Label(num_top_classes=5)

gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch()
