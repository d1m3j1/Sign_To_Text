import os
import cv2
import csv
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
import plotly.express as px 
from tqdm.notebook import tqdm
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split, GroupShuffleSplit 
from IPython.display import HTML

N_ROWS = 543
N_DIMS = 3
DIM_NAMES = ['x', 'y', 'z']
INPUT_SIZE = 64


USE_TYPES = ['left_hand', 'pose', 'right_hand']
LHAND = np.arange(468, 489) # 21
RHAND = np.arange(522, 543) # 21
# POSE  = np.arange(489, 522)# 33
FACE  = np.arange(0,468) #468
LIP = np.array([ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])
LPOSE = np.array([502, 504, 506, 508, 510])
RPOSE = np.array([503, 505, 507, 509, 511])


# Concatenate the relevant landmarks for each dominant hand scenario
left_hand_dominant = np.concatenate((LIP, LHAND, LPOSE))
right_hand_dominant = np.concatenate((LIP, RHAND, RPOSE))
# landmarks = np.concatenate((LIP, RHAND, LHAND, POSE))
hand_index = np.concatenate((LHAND, RHAND), axis = 0)

#Landmark Indices in preprocess data
hands_index = np.argwhere(np.isin(left_hand_dominant, hand_index)).squeeze()
lip_index = np.argwhere(np.isin(left_hand_dominant, LIP)).squeeze()
left_hand_index = np.argwhere(np.isin(left_hand_dominant, LHAND)).squeeze()
right_hand_index = np.argwhere(np.isin(left_hand_dominant, RHAND)).squeeze()
pose_index = np.argwhere(np.isin(left_hand_dominant, LPOSE)).squeeze()
n_cols = left_hand_dominant.size

LIPS_START = 0
LEFT_HAND_START = lip_index.size
RIGHT_HAND_START = LEFT_HAND_START + left_hand_index.size
POSE_START = RIGHT_HAND_START + right_hand_index.size

def pad_edge(t, repeats, side):
    if side == 'LEFT':
        return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
    elif side == 'RIGHT':
        return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)

class TFLitePreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TFLitePreprocessLayer, self).__init__()

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, N_ROWS, N_DIMS], dtype=tf.float32),))
    def call(self, inputData):

        totalFrames = tf.shape(inputData)[0]

        # Calculate the dominant hand based on sum of absolute coordinates
        leftSum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(inputData, LHAND, axis=1)), 0, 1))
        rightSum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(inputData, RHAND, axis=1)), 0, 1))
        isLeftDominant = leftSum >= rightSum

        handIndices = LHAND if isLeftDominant else RHAND
        framesWithHand = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(inputData, handIndices, axis=1)), 0, 1), axis=[1, 2])
        validFrameIndices = tf.squeeze(tf.where(framesWithHand > 0), axis=1)

        processedData = tf.gather(inputData, validFrameIndices, axis=0)
        validFrameIndices = tf.cast(validFrameIndices, tf.float32)
        validFrameIndices -= tf.reduce_min(validFrameIndices)

        frameCount = tf.shape(processedData)[0]

        landmarkColumns = left_hand_dominant if isLeftDominant else right_hand_dominant
        processedData = tf.gather(processedData, landmarkColumns, axis=1)

        # Check if video fits within the input size
        if frameCount < INPUT_SIZE:
            validFrameIndices = tf.pad(validFrameIndices, [[0, INPUT_SIZE - frameCount]], constant_values=-1)
            processedData = tf.pad(processedData, [[0, INPUT_SIZE - frameCount], [0, 0], [0, 0]], constant_values=0)
            processedData = tf.where(tf.math.is_nan(processedData), 0.0, processedData)
            return processedData, validFrameIndices
        else:
            # Handling videos larger than the input size
            if frameCount < INPUT_SIZE ** 2:
                factor = tf.math.floordiv(INPUT_SIZE * INPUT_SIZE, totalFrames)
                processedData = tf.repeat(processedData, repeats=factor, axis=0)
                validFrameIndices = tf.repeat(validFrameIndices, repeats=factor, axis=0)

            adjustedSize = tf.math.floordiv(len(processedData), INPUT_SIZE)
            if tf.math.mod(len(processedData), INPUT_SIZE) > 0:
                adjustedSize += 1

            requiredPadding = (adjustedSize * INPUT_SIZE) - len(processedData) if adjustedSize == 1 else (adjustedSize * INPUT_SIZE) % len(processedData)
            padLeft = tf.math.floordiv(requiredPadding, 2) + tf.math.floordiv(INPUT_SIZE, 2)
            padRight = padLeft + (1 if tf.math.mod(requiredPadding, 2) > 0 else 0)

            processedData = self._addPadding(processedData, padLeft, 'LEFT')
            processedData = self._addPadding(processedData, padRight, 'RIGHT')
            validFrameIndices = self._addPadding(validFrameIndices, padLeft, 'LEFT')
            validFrameIndices = self._addPadding(validFrameIndices, padRight, 'RIGHT')

            processedData = tf.reshape(processedData, [INPUT_SIZE, -1, n_cols, N_DIMS])
            validFrameIndices = tf.reshape(validFrameIndices, [INPUT_SIZE, -1])
            
            processedData = tf.experimental.numpy.nanmean(processedData, axis=1)
            validFrameIndices = tf.experimental.numpy.nanmean(validFrameIndices, axis=1)

            processedData = tf.where(tf.math.is_nan(processedData), 0.0, processedData)
            return processedData, validFrameIndices

    def _addPadding(self, tensor, paddingSize, direction):
        if direction == 'LEFT':
            return tf.concat([tf.repeat(tensor[:1], repeats=paddingSize, axis=0), tensor], axis=0)
        else:
            return tf.concat([tensor, tf.repeat(tensor[-1:], repeats=paddingSize, axis=0)], axis=0)

layerInstance = TFLitePreprocessLayer()

