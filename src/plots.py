import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_average_importance_scores(avg_scores):
  fig, ax = plt.subplots(figsize=(10, 5))
  x = range(len(avg_scores))
  ax.set_xlabel('frame index')
  ax.set_ylabel('average importance scores')
  ax.set_title(title)
  ax.plot(x, avg_scores_test)

def plot_silhoutte_scores(sil_scores):
  fig, ax = plt.subplots()
  ax.plot(range(2, 20), sil_scores[2:20], 'o-')
  ax.set_xlabel("number of clusters")
  ax.set_ylabel("silhouette score")

def rgb_histogram(image_path):
  img = cv2.imread(image_path)
  color = ('b', 'g', 'r')
  for i, col in enumerate(color):
      histr = cv2.calcHist([img], [i], None, [256], [0, 256])
      plt.plot(histr, color=col)
      plt.xlim([0, 256])
  plt.show()
