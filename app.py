import cv2
import copy
import mediapipe as mp
import numpy as np

from utils.normalize_data import normalize_lendmarks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Controladores
# =============================================================================
show_landmarks = True;
# =============================================================================


# Desenha o retangulo em torno da mão
def print_rectangle(img, marks):
  cv2.rectangle(img, (marks[0], marks[1]), (marks[2], marks[3]), (0,0,0), 1);
  return img

# calcula os vertices do retangulo em torno da mão
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# desenha os pontos da mão na tela
def print_landmarks(debug_img, hand_landmarks):
  """
    Desenha os pontos da mão na tela
    Args:
      debug_img (cap.read): imagem para desenhar os pontos
      hand_landmarks (LandmarkList): pontos a serem desenhados na imagem
  """
  if not show_landmarks:
    return
  mp_drawing.draw_landmarks(
            debug_img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=2,
    min_tracking_confidence=0.5) as hands:
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image.flags.writeable = False
    debug_img = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = hands.process(image)
for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
    if results.multi_hand_landmarks:
      for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
        print(handedness)
        print_landmarks(debug_img, hand_landmarks)
        #normalize_lendmarks(hand_landmarks)
        debug_img = print_rectangle(debug_img,calc_bounding_rect(debug_img ,hand_landmarks))

    cv2.imshow('MediaPipe Hands', debug_img)
    key = cv2.waitKey(1) & 0xFF
    match key:
      case 32:
        show_landmarks = not show_landmarks
        print(show_landmarks)
      case 27:
        break 
cap.release()