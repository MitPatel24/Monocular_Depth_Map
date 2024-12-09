import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import timm

# download hte midas
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas.to('cpu')
midas.eval()

# input transformational pipeline
transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.small_transform

def depth_to_distance(depth_value,depth_scale):
  return -1.0/(depth_value*depth_scale)

# cap= cv2.VideoCapture('C:\\Sem-7\\Stereo vision\\davis_dolphins.mp4') #C:\\Sem-7\\Stereo vision\\davis_dolphins.mp4, C:\\Sem-7\\Stereo vision\\ferris_wheel.mp4
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame =cap.read()

    # Transform input for midas
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(400,400))
    imgbatch=transform(img).to('cpu')

    # make prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output=prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        output_norm = (output_norm*255).astype(np.uint8)
        output_norm = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

        # print(output)
    # plt.title('Depth Estimation')
    # plt.axis('off')
    # plt.subplot(1,3,1)
    # plt.figure(figsize=(400,400))
    # plt.title('original')
    # plt.axis('off')
    # plt.imshow(frame)
    # plt.subplot(1,3,2)
    # plt.title('output')
    # plt.axis('off')
    # plt.imshow(output)
    # plt.subplot(1,3,3)
    # plt.title('colored depth map')
    # plt.axis('off')
    # plt.imshow(output_norm)
    
    cv2.imshow('CV2Frame',frame)
    cv2.imshow('Output',output) 
    cv2.imshow('Output_norm',output_norm)
    plt.pause(0.0001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
plt.show()