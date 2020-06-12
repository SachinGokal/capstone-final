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
                             scores=average_scores[average_scores['filename'] ==
                                                   video_id]['average_score'].values[0].reshape(-1, 1)
                             if y == 'first':
                             y=scores
                             else:
                             y=np.concatenate((y, scores), axis=0)
                             print(f'complete for {video_id}')
                             print(np.array(X).shape, np.array(y).shape)
                             X=MinMaxScaler(feature_range=(0, 1)
                                            ).fit_transform(X)
                             return X, y

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

#### General Model Train / Test videos used

## Train (18)

# sTEELN-vY30 BBC - Train crash 2013
# uGu_10sucQo A Year of Beekeeping
# JKpqYvAdIsw ICC World Twenty20 Bangladesh 2014 Flash Mob - Pabna University of Science & Technology ( PUST )
# GsAD1KT1xo8 Parkour Camp Leipzig
# kLxoNp-UchI Oliver's Show - Dog's tale
# eQu1rNs0an0 How to stop your Fixie
# LRw_obCPUt0 GoogaMooga Sneak Peek Joseph Leonard's Fried Chicken Sandwich cooking video
# qqR6AEXwxoQ Motocross Tips & Tricks : How to Whip a Motocross Bike
# jcoYJXDG9sw TODAY- Obie the obese dog works toward weight loss
# gzDbaEs1Rlg ŠKODA Tips How to Repair Your Tyre
# fWutDQy1nnY Chinatown Parade
# _xMr-HKMfVA CASACL - Flashmob in Copenhagen underground - Peer Gynt
# NyBmCxDoHJU The Dog Show HD 720p
# Hl-__g2gn_A Reuben Sandwich with Corned Beef & Sauerkraut
# Bhxk-O1Y7Ho Vlog #509 I'M A PUPPY DOG GROOMER! September 13, 2014
# oDXZc0tZe04 Apis Mellifera in a Vertical Log Hive
# i3wAGJaaktw Pet Joy Spa Grooming Services | Brentwood, CA (310) 471-0088
# 98MoyGZKHXc How to use a tyre repair kit - Which? guide

## Test (10)

# xwqBXPGE9pQ Smart Electric Vehicle Balances on Two Wheels
# XkqCExn6_Us Singapore Parkour Free Running | JC Boy Late for School
# z_6gVvQb2d0 LA Kings Stanley Cup South Bay Parade 2014
# VuWGsYPqAX8 Flash mob protest for Syria | Sydney, Australia, May 2012 | #Silenceisbetrayal
# xxdtq8mxegs How to Clean Your Dog's Ears - Vetoquinol USA
# xmEERLqJ2kU Bluffton Teachers Flash Mob Dance (A Staff that Cares About Their Students)
# XzYM3PfTM4w When to Replace Your Tires GMC
# WG0MBPpPC6I Mexican Fried Chicken Sandwich Recipe
# Yi4Ij2NM7U4 Poor Man's Meals: Spicy Sausage Sandwich
# WxtbjNsCQ8A Beekeeping101.mov

#### Category Model Train / Test Videos Bike Tricks, Parkour, and Vehicles categories (4 Train, 1 Test)

### Bike Tricks Category

## Train

# eQu1rNs0an0 How to stop your Fixie
# EYqVtI9YWJA Smage Bros. Motorcycle Stunt Show
# JgHubY5Vw3Y How to lock your bike. The RIGHT way!
# iVt07TCkFM0 Pure Fix TV: How to Wheelie

## Test

# qqR6AEXwxoQ Motocross Tips & Tricks: How to Whip a Motocross Bike

### Parkour Category

## Train

# GsAD1KT1xo8 Parkour Camp Leipzig
# b626MiF1ew4 Charlotte Parkour | Charlotte Video Project
# PJrm840pAUI Vivencias: Jam Parkour Viña del Mar 2012
# cjibtmSLxQ4 David Belle | Fondateur du parkour | Reportage de TF1

## Test

# XkqCExn6_Us Singapore Parkour Free Running | JC Boy Late for School

### Vehicles Category

## Train

# sTEELN-vY30 BBC - Train crash 2013
# xwqBXPGE9pQ Smart Electric Vehicle Balances on Two Wheels
# HT5vyqe0Xaw The stuck truck of Mark, The rut that filled an afternoon.
# vdmoEJ5YbrQ  # 453 girl gets van stuck in the back fourty [Davidsfarm]

## Test

# akI8YFjEmUw Electric cars making earth more green
