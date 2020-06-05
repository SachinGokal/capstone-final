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

def create_train_or_test_data_set_for_vgg(data_set='train'):
  X = 'first'
  y = 'first'
  for directory in glob.glob(f'data/frames/{data_set}/*'):
      frames = glob.glob(f'{directory}/*')
      video_id = directory.split(f'data/frames/{data_set}/')[1]
      sorted_frames = sort_frames(frames, video_id, data_set=data_set)
      if X == 'first':
          X = get_vgg_features(sorted_frames)
      else:
          X = np.concatenate((X, get_vgg_features(sorted_frames))
      scores = average_scores[average_scores['filename'] ==
                              video_id]['average_score'].values[0].reshape(-1, 1)
      if y == 'first':
          y = scores
      else:
          y = np.concatenate((y, scores), axis=0)
      print(f'complete for {video_id}')
      print(np.array(X).shape, np.array(y).shape)
  X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
  return X, y

def rgb_features(image_list, bins):
    features = []
    for frame in image_list:
      cv_img = cv2.imread(frame)
      histr = cv2.calcHist([cv_img], [0, 1, 2], None, [
                           bins, bins, bins], [0, 256, 0, 256, 0, 256])
      cv2.normalize(hist, hist)
      features.append(histr.flatten())
    return features

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

# Figure out which video is missing a missing frame score
def create_train_or_test_data_set_for_rgb(data_set='train'):
  X = 'first'
  y = 'first'
  for directory in glob.glob(f'data/frames/{data_set}/*'):
      frames = glob.glob(f'{directory}/*')
      video_id = directory.split(f'data/frames/{data_set}/')[1]
      sorted_frames = sort_frames(frames, video_id, data_set=data_set)
      if X == 'first':
          X = rgb_features(sorted_frames, 8)
      else:
          X = np.concatenate(
              (X, rgb_features(sorted_frames, 8)), axis=0)
      scores = average_scores[average_scores['filename'] ==
                              video_id]['average_score'].values[0].reshape(-1, 1)
      if y == 'first':
          y = scores
      else:
          y = np.concatenate((y, scores), axis=0)
      print(f'complete for {video_id}')
      print(np.array(X).shape, np.array(y).shape)
  X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
  return X, y
  # Figure out which video is missing a missing frame score

def create_train_or_test_data_set_for_hsv(data_set='train'):
  X = 'first'
  y = 'first'
  for directory in glob.glob(f'data/frames/{data_set}/*'):
      frames = glob.glob(f'{directory}/*')
      video_id = directory.split(f'data/frames/{data_set}/')[1]
      sorted_frames = sort_frames(frames, video_id, data_set=data_set)
      if X == 'first':
          X = hsv_features(sorted_frames, 8)
      else:
          X = np.concatenate(
              (X, rgb_features(sorted_frames, 8)), axis=0)
      scores = average_scores[average_scores['filename'] ==
                              video_id]['average_score'].values[0].reshape(-1, 1)
      if y == 'first':
          y = scores
      else:
          y = np.concatenate((y, scores), axis=0)
      print(f'complete for {video_id}')
      print(np.array(X).shape, np.array(y).shape)
  X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
  return X, y

def create_single_video_data_set_for_rgb(data_set='train', directory):
    X = 'first'
    y = 'first'
    frames = glob.glob(f'{directory}/*')
    video_id = directory.split(f'data/frames/{data_set}/')[1]
    sorted_frames = sort_frames(frames, video_id, data_set=data_set)
    if X == 'first':
        X = rgb_features(sorted_frames, 8)
    else:
        X = np.concatenate(
            (X, rgb_features(sorted_frames, 8)), axis=0)
    scores = average_scores[average_scores['filename'] ==
                            video_id]['average_score'].values[0].reshape(-1, 1)
    if y == 'first':
        y = scores
    else:
        y = np.concatenate((y, scores), axis=0)
    print(f'complete for {video_id}')
    print(np.array(X).shape, np.array(y).shape)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    return X, y
