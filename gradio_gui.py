from keras.models import load_model
from keras.layers import Rescaling,Resizing
import tensorflow as tf
import numpy as np
import gradio as gr
from numpy import asarray


model = load_model('./checkpoints/checkpoint.model.keras')

scale = Rescaling(1./255)
resize = Resizing(224,224)


def action_recognition(image):

    preds = ['Calling','Clapping','Cycling','Dancing','Drinking','Eating','Fighting',
             'Hugging','Laughing','Listening to Music','Running or Walking','Sitting','Sleeping','Texting','Using Laptop'] 

    # Read image and showy
    img = asarray(image)
    
    # Preprocess image
    img = scale(img)
    img = resize(img)
    img = tf.reshape(img,(1,224,224,3))

    # prediction
    pred = model.predict(img)

    # Mapping indices to their respective class labels
    if np.argmax(pred) == 0:
        print('Calling')
    elif np.argmax(pred) == 1:
        print('Clapping')
    elif np.argmax(pred) == 2:
        print('Cycling')
    elif np.argmax(pred) == 3:
        print('Dancing')
    elif np.argmax(pred) == 4:
        print('Drinking')
    elif np.argmax(pred) == 5:
        print('Eating')
    elif np.argmax(pred) == 6:
        print('Fighting')
    elif np.argmax(pred) == 7:
        print('Hugging')
    elif np.argmax(pred) == 8:
        print('Laughing')
    elif np.argmax(pred) == 9:
        print('Listening to Music')
    elif np.argmax(pred) == 10:
        print('Running')
    elif np.argmax(pred) == 11:
        print('Sitting')
    elif np.argmax(pred) == 12:
        print('Sleeping')
    elif np.argmax(pred) == 13:
        print('Texting')
    elif np.argmax(pred) == 14:
        print('Using Laptop')
    
    # Return the predicted class index and prediction array
    return preds[np.argmax(pred)]
    

demo = gr.Interface(
    fn=action_recognition,
    inputs=[gr.Image(label="Image")],
    outputs=['text'],
    allow_flagging='never'
)



