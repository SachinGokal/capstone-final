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
