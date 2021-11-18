import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

mp_selfie = mp.solutions.selfie_segmentation

def segment(image): 
    with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
        res = model.process(image)
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5 
        return np.where(mask, image, cv2.blur(image, (40,40)))

webcam = gr.inputs.Image(shape=(640, 480), source="webcam")
webapp = gr.interface.Interface(fn=segment, inputs=webcam, outputs="image", allow_flagging=False)
webapp.launch()