import os
import cv2
import glob
import numpy as np
import pandas as pd


def create_video_summary(model, frames, video_as_features, time_length):
  scores = model.predict(video_as_features)
  scores_median = np.median(scores)
  impt_frames = np.argwhere(scores, scores > scores_median)
  return create_video_from_frames(impt_frames)

def create_video_from_frames(frames, title, fps=24, fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v')):
    img_array = []
    for filename in frames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(title, fourcc, fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
