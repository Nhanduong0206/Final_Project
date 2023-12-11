import cv2
import numpy as np
import pygame as pg
import pickle
import torch
import torchvision.transforms as transforms
import time
from PIL import Image




class PretrainModel:
    def __init__(self):
        self.model = torch.load("models/pretrained.pkl")
        self.model.eval()
        self.transformer =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_feature(self, pil_image):
        image_tensor = self.transformer(pil_image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            feature = self.model(image_tensor).squeeze().detach().cpu().numpy()
        return feature
    
Car_Detector = pickle.load(open('model.pickle', 'rb'))    
pretrainmodel= PretrainModel()
pg.mixer.init()
pg.mixer.music.load("warning.mp3")
cam = cv2.VideoCapture('test.mp4')
count = 0

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = cam.read() 
    
    
    # Display the resulting frame 
    
    
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_rate = int(cam.get(cv2.CAP_PROP_FPS))

    

    roi = frame[frame_height//6:5* frame_height//6,frame_width//6:5*frame_width//6]
    ROI = Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
    
    
    if Car_Detector.predict(pretrainmodel.get_feature(ROI).reshape(1,-1)) == 1:
        rect_red = cv2.rectangle(frame, (frame_width//6,frame_height//6), (5*frame_width//6,5* frame_height//6), (0,0 , 255), 1)
        count += 1
        if count >= 10:
            pg.mixer.music.play()
        
    else:
        rect_green = cv2.rectangle(frame, (frame_width//6,frame_height//6), (5*frame_width//6,5* frame_height//6), (0, 255, 0), 1)
        count = 0

    
    
    cv2.imshow('frame', frame) 
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice )
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
cam.release() 
# Destroy all the windows 
cv2.destroyAllWindows()

