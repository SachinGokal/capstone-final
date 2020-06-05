import os
import cv2
import glob
import numpy as np
import pandas as pd

def info_df():
  return pd.read_csv('data/annotations.csv', header=None)

def annotations_df():
  annotations = pd.read_csv('data/annotations.csv', header=None)
  annotations.rename(
      columns={0: 'filename', 1: 'category', 2: 'annotations'}, inplace=True)
  annotations['annotations'] = annotations['annotations'].apply(
    lambda a: a.split(','))
  return annotations

def average_scores_df(annotations=annotations_df()):
  avg_annotations = []
  for fname in annotations.filename.unique():
      values = annotations[annotations['filename']
                          == fname]['annotations'].values
      float_conversion = np.array(list(values)).astype('float64')
      avg = np.mean(float_conversion, axis=0)
      avg_annotations.append(avg)
  average_scores = pd.DataFrame(
      {'filename': annotations.filename.unique(), 'average_score': avg_annotations})
  return average_scores

def full_df(info=info_df(), average_scores=average_scores_df()):
  full_df = pd.concat([info.set_index('video_id'), average_scores.set_index(
      'filename')], axis=1, join='inner')
  full_df = full_df.reset_index().rename(columns={'index': 'video_id'})
  return full_df

def create_frames(data_set='train'):
  for video in glob.glob('data/video/*.mp4'):
    video_id = video.split('data/video/')[1].split('.mp4')[0]
    video_capture = cv2.VideoCapture(video)
    currentframe = 0
    try:
        if not os.path.exists(f'data/frames/{data_set}/{video_id}'):
            os.makedirs(f'data/frames/{data_set}/{video_id}')
    except OSError:
        print('Error: Creating directory of data')
    while(True):
        ret, frame = video_capture.read()
        if ret:
            name = f'data/frames/{data_set}/{video_id}/' + str(currentframe) + '.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    video1_capture.release()
    cv2.destroyAllWindows()

def sort_frames(list_of_frames, video_id, data_set='train'):
    return sorted(list_of_frames, key=lambda x: int(x.split(f'data/frames/{data_set}/{video_id}/')[1].split('.jpg')[0]))


