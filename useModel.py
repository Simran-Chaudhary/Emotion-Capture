import sys
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
# parameters for loading data and images
image_path = sys.argv[1]
emotion_model_path = 'model1.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')
gray_image = cv2.resize(gray_image, (emotion_target_size))
gray_image = np.expand_dims(gray_image, 0)
gray_image = np.expand_dims(gray_image, -1)
emotion_label_arg = np.argmax(emotion_classifier.predict(gray_image))
emotion_text = emotion_labels[emotion_label_arg]
print(emotion_text)
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('predicted_test_image.png', bgr_image)