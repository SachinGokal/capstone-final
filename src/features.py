import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# vgg features
def get_vgg_features(image_list):
  image_features = []
  base_model = VGG16(weights='imagenet', include_top=False)
  model = Model(inputs=base_model.input,
                outputs=base_model.get_layer('block5_conv3').output)

  for frame in image_list:
    img = image.load_img(frame, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)
    image_features.append(np.array(vgg16_feature).flatten())

  return image_features

# rgb features
def rgb_features(image_list, bins):
    features = []
    for frame in image_list:
      cv_img = cv2.imread(frame)
      histr = cv2.calcHist([cv_img], [0, 1, 2], None, [
                           bins, bins, bins], [0, 256, 0, 256, 0, 256])
      cv2.normalize(hist, hist)
      features.append(histr.flatten())
    return features

# hsv features
def hsv_features(image_list, bins):
    features = []
    for frame in image_list:
      cv_img = cv2.imread(frame)
      img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
      histr = cv2.calcHist([cv_img], [0, 1, 2], None, [
                           bins, bins, bins], [0, 256, 0, 256, 0, 256])
      cv2.normalize(hist, hist)
      features.append(histr.flatten())
    return features
