import cv2
import sys
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
from collections import deque
import json
from sklearn.preprocessing import LabelEncoder
from test_1 import TFLitePreprocessLayer
layerInstance = TFLitePreprocessLayer()

NUM_CLASSES = 250
optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)

ROWS_PER_FRAME = 543  # number of landmarks per frame

def scce_with_ls(y_true, y_pred):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)

with open('asl-signs/sign_to_prediction_index_map.json') as f: 
    sign_map = json.load(f)

def mapping_sign_code(sign: str):
    return sign_map[sign]

signs = list(sign_map)
train = pd.read_csv('asl-signs/train.csv')
train['sign_code'] = train['sign'].apply(mapping_sign_code)

# Instantiate the encoder
le = LabelEncoder()

# Fit the encoder and transform the 'sign' column
train['sign_code'] = le.fit_transform(train['sign'])

# Create dictionaries for mapping
SIGN2ORD = dict(zip(le.classes_, le.transform(le.classes_)))
ORD2SIGN = dict(zip(le.transform(le.classes_), le.classes_))

#Label
labels = [ORD2SIGN.get(i).replace(' ', '_') for i in range(NUM_CLASSES)]
custom_objects = {
    'scce_with_ls': scce_with_ls
}

# Load the transformer model
transformer_model = tf.keras.models.load_model("model/full_transformer_model", custom_objects=custom_objects)
lstm_model = tf.keras.models.load_model("model/full_lstm_model", custom_objects=custom_objects)


def insert_missing_landmarks(frame_number, hand_type, num_landmarks=21):
    """Helper function to insert NaN values for missing landmarks"""
    missing_data = []
    for _ in range(num_landmarks):
        missing_data.append({'frame': frame_number, 'type': hand_type, 'x': float('nan'), 'y': float('nan'), 'z': float('nan')})
    return missing_data

def dataframe_to_array(df):
    data_columns = ['x', 'y', 'z']
    relevant_data = df[data_columns]
    n_frames = int(len(relevant_data) / ROWS_PER_FRAME)
      
    data_array = relevant_data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))

    print(f'data_array shape: {data_array.shape}')

    return data_array.astype(np.float32)    

def predict_signs(fram, non_empty_frame, model):    
    """Predict signs using the provided model."""
    predictions = model.predict({'frames': fram, 'non_empty_frame_idxs': non_empty_frame}, verbose=2)
    return predictions

def main(video_path, model):
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands()
    face = mp_face.FaceMesh()
    pose = mp_pose.Pose()
    WINDOW_SIZE = 50
    cap = cv2.VideoCapture(video_path)

    landmarks_storage = deque(maxlen = WINDOW_SIZE)  # Store landmarks for each frame here
    last_prediction = None
    last_valid_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if last_valid_frame is not None:
                if last_prediction:
                    cv2.putText(last_valid_frame, f"Prediction: {last_prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Live Prediction', last_valid_frame)  # Show the last frame
                cv2.waitKey(10000)  # Wait for 10 seconds
            break
        last_valid_frame = frame
        # Convert the BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        face_results = face.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        frame_landmarks = []  # Store landmarks for this particular frame

        # Extract and Draw Hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmark in hand_results.multi_hand_landmarks:
                frame_landmarks.extend([(lm.x, lm.y, lm.z) for lm in hand_landmark.landmark])
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

        # Extract and Draw Face landmarks
        if face_results.multi_face_landmarks:
            for face_landmark in face_results.multi_face_landmarks:
                frame_landmarks.extend([(lm.x, lm.y, lm.z) for lm in face_landmark.landmark])
                mp.solutions.drawing_utils.draw_landmarks(frame, face_landmark)

        # Extract and Draw Pose landmarks
        if pose_results.pose_landmarks:
            pose_landmark = pose_results.pose_landmarks
            frame_landmarks.extend([(lm.x, lm.y, lm.z) for lm in pose_landmark.landmark])
            mp.solutions.drawing_utils.draw_landmarks(frame, pose_landmark, mp_pose.POSE_CONNECTIONS)

        # Assuming the combined landmarks for hands, face, and pose is 543
        if len(frame_landmarks) == 543:
            landmarks_storage.append(frame_landmarks)

            if len(landmarks_storage) == WINDOW_SIZE:
                landmarks_data = np.array(landmarks_storage)
                landmarks_data = np.array(landmarks_storage)
                frames, non_empty_frames = layerInstance(landmarks_data)
                frames = np.expand_dims(frames, axis=0)
                non_empty_frames = np.expand_dims(non_empty_frames, axis=0)
                raw_output = predict_signs(frames, non_empty_frames, model)
                pred_idx = raw_output.argmax(axis = 1)
                confidence = raw_output[0][pred_idx[0]]
                print(confidence)
                sign = ORD2SIGN[pred_idx[0]]                
                last_prediction = f'{sign} (Confidence Level : {confidence:.2f})'

                # Display prediction on video
        if last_prediction:
            cv2.putText(frame, f"Prediction: {last_prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Display the video frame with landmarks and prediction
        cv2.imshow('Live Prediction', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = sys.argv[1]
    main(video_path, transformer_model)