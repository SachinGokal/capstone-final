import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.style.use('ggplot')

def plot_average_importance_scores(avg_scores, title):
  fig, ax = plt.subplots(figsize=(10, 5))
  x = range(len(avg_scores))
  ax.set_xlabel('frame index')
  ax.set_ylabel('average importance scores')
  ax.set_title(title)
  ax.plot(x, avg_scores)

def plot_peak_frames(quantile=0.75, scores, images):
  peaks, _ = find_peaks(
      scores, height=np.quantile(scores, quantile))
  peak_frames = np.array(images)[peaks]
  fig, axs = plt.subplots(10, 1, figsize=(30, 30))
  for frame, ax in zip(peak_frames, axs.flatten()):
      ax.imshow(plt.imread(frame))

def plot_silhoutte_scores(sil_scores):
  fig, ax = plt.subplots()
  ax.plot(range(2, 20), sil_scores[2:20], 'o-')
  ax.set_xlabel("number of clusters")
  ax.set_ylabel("silhouette score")

def plot_rgb(image):
  img = cv2.imread(image)
  color = ('b', 'g', 'r')
  for i, col in enumerate(color):
      histr = cv2.calcHist([img], [i], None, [256], [0, 256])
      plt.plot(histr, color=col)
      plt.xlim([0, 256])
  plt.show()
