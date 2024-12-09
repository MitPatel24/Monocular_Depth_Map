import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

# Load the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

def depth_to_distance(depth_value, depth_scale):
    return -1.0 / (depth_value * depth_scale)

# Load the image
image_path = 'C:\Sem-7\Stereo vision\my.jpg'  # Replace with your image path
img = cv2.imread(image_path)

# Transform input for MiDaS
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (400, 400))
img_batch = transform(img).to('cpu') #img_resized

# Make prediction
with torch.no_grad():
    prediction = midas(img_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode='bicubic',
        align_corners=False
    ).squeeze()

    output = prediction.cpu().numpy()
    output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    output_norm = (output_norm * 255).astype(np.uint8)
    output_norm_colored = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

# Display results
plt.suptitle('Results of MiDaS',fontsize=14)
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(img_rgb)

plt.subplot(1, 3, 2)
plt.title('Depth Map')
plt.axis('off')
plt.imshow(output, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Depth Map')
plt.axis('off')
plt.imshow(output, cmap='inferno')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the main title
plt.show()

plt.subplot(1, 3, 3)
plt.title('Colored Depth Map')
plt.axis('off')
plt.imshow(output_norm_colored)

# for saving resultant images
# plt.imshow(output, cmap='gray')
# plt.axis('off')
# plt.savefig('MiDaS_Gray.jpg', format='jpg', dpi=500)