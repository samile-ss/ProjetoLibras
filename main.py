import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque


CONFIDENCE_THRESHOLD = 0.60  
SMOOTHING_FRAMES = 10        
OFFSET = 30                  

np.set_printoptions(suppress=True)

print("A carregar o modelo...")
model = load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as file:
    raw_labels = [line.strip() for line in file.readlines()]


def parse_label(raw: str) -> str:
    parts = raw.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1].strip()
    return raw.strip()

class_names = [parse_label(l) for l in raw_labels]
print(f"Classes carregadas: {class_names}")


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)

def get_smoothed_prediction(new_index: int, new_confidence: float):
    """Adiciona predição ao buffer e retorna a classe mais votada."""
    prediction_buffer.append(new_index)
    # Conta votos
    votes = {}
    for idx in prediction_buffer:
        votes[idx] = votes.get(idx, 0) + 1
    best_index = max(votes, key=votes.get)
    return best_index


def prepare_image(clean_frame: np.ndarray, x_min: int, y_min: int,
                  x_max: int, y_max: int) -> np.ndarray:
    
    """Corta a região da mão, redimensiona e normaliza para o modelo."""

    img_crop = clean_frame[y_min:y_max, x_min:x_max]

    canvas_size = 224
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    crop_h, crop_w = img_crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return None

    scale = canvas_size / max(crop_h, crop_w)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (canvas_size - new_h) // 2
    left = (canvas_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized

    # Normaliza para o Teachable Machine: [-1, 1]
    img_array = np.asarray(canvas, dtype=np.float32)
    normalized = (img_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)  # shape (1, 224, 224, 3)


cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")

with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,           
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.flip(frame, 1)


        clean_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Esqueleto no frame de exibição
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

                x_min = max(0, min(x_coords) - OFFSET)
                y_min = max(0, min(y_coords) - OFFSET)
                x_max = min(w, max(x_coords) + OFFSET)
                y_max = min(h, max(y_coords) + OFFSET)

                # Retângulo verde ao redor da mão
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Preparar imagem usando o frame LIMPO
                data = prepare_image(clean_frame, x_min, y_min, x_max, y_max)

                if data is not None:
                    prediction = model.predict(data, verbose=0)
                    raw_index = int(np.argmax(prediction))
                    confidence = float(prediction[0][raw_index])

                    if confidence > CONFIDENCE_THRESHOLD:
                        # Suavização temporal
                        smooth_index = get_smoothed_prediction(raw_index, confidence)
                        label_text = class_names[smooth_index]
                        text = f"{label_text}  {round(confidence * 100)}%"

                        
                        (text_w, text_h), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                        text_x = x_min
                        text_y = max(y_min - 10, text_h + 10)

                        # Fundo escuro para o texto
                        cv2.rectangle(frame,
                                      (text_x - 4, text_y - text_h - 6),
                                      (text_x + text_w + 4, text_y + 4),
                                      (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame, text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0), 2)
                    else:
                        prediction_buffer.clear()  

        # Exibir instruções no canto
        cv2.putText(frame, "Pressione 'q' para sair",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        cv2.imshow('Reconhecimento de Libras', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()