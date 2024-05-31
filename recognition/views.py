import os
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 피클 파일 경로 설정
model_path = 'C:/model/word_rf_model.pkl'
label_encoder_path = 'C:/pkl/word_label_encoder.pkl'

# 모델과 라벨 인코더 로드
with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def index(request):
    return render(request, 'recognition/index.html')

def calculate_scores(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def predict_sign(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body = json.loads(body_unicode)
            landmarks = np.array(body['landmarks']).flatten()

            # Ensure the landmarks array has 126 elements (42 landmarks * 3 coordinates)
            if len(landmarks) == 63:  # If only one hand is detected
                landmarks = np.concatenate((landmarks, np.zeros(63)))

            landmarks = landmarks.reshape(1, -1)

            prediction_probs = model.predict_proba(landmarks)
            top_indices = np.argsort(prediction_probs[0])[-2:][::-1]
            top_classes = label_encoder.inverse_transform(top_indices)
            top_probs = prediction_probs[0][top_indices]

            predicted_label = label_encoder.inverse_transform([model.predict(landmarks)[0]])[0]

            # 가중치 설정
            weights = {'precision': 0.25, 'recall': 0.25, 'f1': 0.25, 'accuracy': 0.25}

            # 실제 라벨과 예측 라벨을 같다고 가정 (실제 사용 시 실제 라벨 필요)
            y_true = [predicted_label]
            y_pred = [predicted_label]

            precision, recall, f1, accuracy = calculate_scores(y_true, y_pred)
            combined_score = (weights['precision'] * precision + 
                              weights['recall'] * recall + 
                              weights['f1'] * f1 + 
                              weights['accuracy'] * accuracy)

            final_prediction = predicted_label if combined_score > 0.5 else 'Uncertain'

            return JsonResponse({
                'classes': [str(cls) for cls in top_classes],
                'probabilities': [float(prob) for prob in top_probs],
                'predicted_label': str(predicted_label),
                'combined_score': float(combined_score),
                'final_prediction': str(final_prediction)
            })
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            print(f"Error during prediction: {error_message}")
            return JsonResponse({'error': 'Prediction error', 'details': error_message}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)