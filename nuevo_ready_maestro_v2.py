import cv2
import mediapipe as mp
from numba import njit
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import os
import socket
from multiprocessing import Lock

# Configuración TCP
TCP_IP = "127.0.0.1"
TCP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))

def send_gesture(msg):
    sock.sendall(msg.encode())
    print(f"TCP >> {msg}")

# Esto queda aquí porque dentro de main no funciona :(
m_estado_orquesta = Lock()
estado_orquesta = "IDLE"

def main():
    model_path_hand = "hand_landmarker.task"
    model_path_pose = "pose_landmarker.task"

    HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
                        (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
                        (0,17), (17,18), (18,19), (19,20)]

    # General var
    ALPHA = 0.65
    m_prev_hands = Lock()
    prev_hands = {}
    m_historial_pos = Lock()
    historial_pos = {0: [], 1: []}

    # Pose var
    m_altura_pecho_y = Lock()
    altura_pecho_y = 0.4

    m_altura_cadera_y = Lock()
    altura_cadera_y = 0.8

    def process_hands(result_hand: HandLandmarkerResult, mp_image: image_lib.Image, timestamp_ms: int):
        # Medo (va a buscar la variable global y deja modificarla)
        global estado_orquesta
        # --- PROCESAMIENTO DE MANOS ---
        if result_hand.hand_landmarks:
            for h_idx, hand in enumerate(result_hand.hand_landmarks):
                label = result_hand.handedness[h_idx][0].category_name
                mano_nombre = "DERECHA" if label == "Left" else "IZQUIERDA"

                # Suavizado EMA
                current_smoothed = []
                if h_idx in prev_hands:
                    for i, lm in enumerate(hand):
                        prev = prev_hands[h_idx][i]
                        current_smoothed.append((ALPHA*lm.x + (1-ALPHA)*prev[0], ALPHA*lm.y + (1-ALPHA)*prev[1]))
                else:
                    current_smoothed = [(lm.x, lm.y) for lm in hand]
                prev_hands[h_idx] = current_smoothed

                # Lógica de ZONA MEDIA (READY) - Mantenida según tu petición
                muneca_y = current_smoothed[0][1]

                en_zona_media = False
                with m_altura_pecho_y:
                    with m_altura_cadera_y:
                        en_zona_media = altura_pecho_y < muneca_y < altura_cadera_y
                        bajo_la_cadera = muneca_y > altura_cadera_y

                with m_estado_orquesta:
                    if en_zona_media and estado_orquesta == "IDLE":
                        estado_orquesta = "READY"
                        send_gesture("READY")
                    # 2. De STOP a IDLE (Bajar manos por debajo de la cadera)
                    elif bajo_la_cadera and estado_orquesta == "STOP":
                        estado_orquesta = "IDLE"
                        send_gesture("IDLE")

                # --- DETECCIÓN DE SUBIDA OPTIMIZADA (START / VOLUME) ---
                # Usamos el dedo medio (punto 12) como en tu código original
                if mano_nombre == "DERECHA":
                    dedo_y = current_smoothed[12][1]
                    historial_pos[h_idx].append((dedo_y, time.time()))

                    # Reducimos el historial a 8 para más velocidad de respuesta
                    if len(historial_pos[h_idx]) > 8: historial_pos[h_idx].pop(0)

                    if len(historial_pos[h_idx]) >= 5:
                        # Calculamos la subida (valor inicial Y - valor final Y)
                        # En MediaPipe, subir es que Y disminuya, por eso inicial - final
                        subida = historial_pos[h_idx][0][0] - historial_pos[h_idx][-1][0]

                        # START: Sensibilidad alta (0.06 es suficiente para un latigazo)
                        with m_estado_orquesta:
                            if estado_orquesta == "READY" and subida > 0.06:
                                estado_orquesta = "PLAYING"
                                send_gesture("START")
                                historial_pos[h_idx] = [] # Limpiar para evitar doble disparo

                            elif estado_orquesta == "PLAYING":
                                x_pulgar = current_smoothed[4][0]
                                y_pulgar = current_smoothed[4][1]

                                x_indice = current_smoothed[8][0]
                                y_indice = current_smoothed[8][1]

                                # distancia de vectores
                                d_pulgar_indice = math.sqrt(((x_pulgar - x_indice)**2) + ((y_pulgar - y_indice)**2))

                                if d_pulgar_indice < 0.03:
                                    estado_orquesta = "STOP"
                                    send_gesture("STOP")
                                    historial_pos[h_idx] = [] # Limpiar para evitar doble disparo


                # # Dibujo básico
                # color = (0, 255, 0) if en_zona_media else (0, 0, 255)
                # for s, e in HAND_CONNECTIONS:
                #     cv2.line(frame, (int(current_smoothed[s][0]*w_img), int(current_smoothed[s][1]*h_img)),
                #              (int(current_smoothed[e][0]*w_img), int(current_smoothed[e][1]*h_img)), color, 2)

    # ... (Configuración de modelos y opciones igual que en tu código original)
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options_hand = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_hand),
        running_mode=VisionRunningMode.LIVE_STREAM, num_hands=2, result_callback=process_hands)
    options_pose = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_pose),
        running_mode=VisionRunningMode.VIDEO)

    detector_hand = mp.tasks.vision.HandLandmarker.create_from_options(options_hand)
    detector_pose = mp.tasks.vision.PoseLandmarker.create_from_options(options_pose)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detector_hand.detect_async(mp_image, timestamp_ms)
        result_pose = detector_pose.detect_for_video(mp_image, timestamp_ms)

        # --- LÓGICA DE POSE (ZONA MEDIA) ---
        with m_altura_pecho_y:
            altura_pecho_y = 0.4

        with m_altura_cadera_y:
            altura_cadera_y = 0.8

        if result_pose.pose_landmarks:
            lm_pose = result_pose.pose_landmarks[0]
            y_hombros = (lm_pose[11].y + lm_pose[12].y) / 2
            y_cadera = (lm_pose[23].y + lm_pose[24].y) / 2
            with m_altura_pecho_y:
                altura_pecho_y = y_hombros + (y_cadera - y_hombros) * 0.35
                cv2.line(frame, (0, int(altura_pecho_y*h_img)), (w_img, int(altura_pecho_y*h_img)), (0, 255, 255), 2)
            with m_altura_cadera_y:
                altura_cadera_y = y_cadera
                cv2.line(frame, (0, int(altura_cadera_y*h_img)), (w_img, int(altura_cadera_y*h_img)), (0, 0, 255), 2)

        with m_estado_orquesta:
            cv2.putText(frame, f"ESTADO: {estado_orquesta}", (10, 50), 2, 1, (255, 255, 255), 2)
        cv2.imshow("Director Console", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
