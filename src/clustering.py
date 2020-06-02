import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.cluster import KMeans, spectral_clustering

def get_image_features(frames_filepath):
  image_features = []
  model = VGG16(weights='imagenet', include_top=False)
  for img_path in glob.glob(frames_filepath)
  img = image.load_img(img_path, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)

  vgg16_feature = model.predict(img_data)
  image_features.append(np.array(vgg16_feature).flatten())
  return image_features

def kmeans(image_features):
  kmeans = KMeans(n_clusters=n, random_state=42).fit(image_features)
  return kmeans

def closest_to_centroid_frames(kmeans, image_list):
  closest, _ = pairwise_distances_argmin_min(
      kmeans.cluster_centers_, image_features)
  np.array(image_list)[closest]
