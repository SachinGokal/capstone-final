import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.cluster import KMeans, spectral_clustering

## Approach #1 with high number of clusters for dynamic with VGG features

# Predict image features using VGG
# Use kmean clustering with n_clusters that represent 10% of the set of the frames
# Choose a frame closest to the centroid for each cluster for constructing summary
# Sort frames to generate the summary

## Approach #2 with low number of clusters for static with RGB features (or VGG)

# Same as above but calculate RGB histogram values for each frame and randomly
# select frame from cluster


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

def kmeans(image_features, n_clusters):
  silhouette = np.zeros(n_clusters)
  for k in range(2, n_clusters):
    kmeans = KMeans(n_clusters=k).fit_predict(image_features)
    silhouette[k] = silhouette_score(image_features, kmeans)
    print(f'{k} cluster complete')
  return silhouette

def closest_to_centroid_frames(kmeans, image_list):
  closest, _ = pairwise_distances_argmin_min(
      kmeans.cluster_centers_, image_features)
  np.array(image_list)[closest]

def random_frame_idxs_for_cluster(kmeans_labels)
  idxs = []
  for label in range(len(np.unique(kmeans_labels))):
    idxs.append(np.random.choice(np.argwhere(
        kmeans_labels == label).flatten(), 1)[0])
  return np.sort(np.array(idxs))

def rgb_histogram_features(image_list):
    features = []
    for frame in image_list:
      img = cv2.imread(frame)
      histr = cv2.calcHist([img],[i],None,[256],[0,256])
      features.append(histr)
    return np.array(features)[:, :, 0]

