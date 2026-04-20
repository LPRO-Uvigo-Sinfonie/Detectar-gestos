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

# Configuracion TCP
TCP_IP = "127.0.0.1"
TCP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))

def send_gesture(msg):
    sock.sendall(msg.encode())
    print(f"TCP >> {msg}")

# Variables Globales
m_estado_orquesta = Lock()
estado_orquesta = "IDLE"

# Variables de Volumen
inerciaVolumen = 8.0
volumenSuavizado = 0.5
last_time_vol = time.time()

def main():
    model_path_hand = "hand_landmarker.task"
    model_path_pose = "pose_landmarker.task"

    # General var
    ALPHA = 0.65
    prev_hands = {}
    historial_pos = {0: [], 1: []}

    # Pose var
    m_altura_pecho_y = Lock()
    altura_pecho_y = 0.4
    m_altura_cadera_y = Lock()
    altura_cadera_y = 0.8

    def process_hands(result_hand, mp_image, timestamp_ms):
        global estado_orquesta, volumenSuavizado, last_time_vol
        
        # Calculo de delta_time para el suavizado del volumen
        current_time = time.time()
        dt = current_time - last_time_vol
        last_time_vol = current_time

        if result_hand.hand_landmarks:
            for h_idx, hand in enumerate(result_hand.hand_landmarks):
                # En MediaPipe Live Stream, 'Left' suele ser la mano derecha fisica y viceversa
                label = result_hand.handedness[h_idx][0].category_name
                mano_nombre = "DERECHA" if label == "Left" else "IZQUIERDA"

                # Suavizado EMA de los puntos
                current_smoothed = []
                if h_idx in prev_hands:
                    for i, lm in enumerate(hand):
                        prev = prev_hands[h_idx][i]
                        current_smoothed.append((ALPHA*lm.x + (1-ALPHA)*prev[0], ALPHA*lm.y + (1-ALPHA)*prev[1]))
                else:
                    current_smoothed = [(lm.x, lm.y) for lm in hand]
                prev_hands[h_idx] = current_smoothed

                # --- LoGICA DE VOLUMEN (MANO IZQUIERDA) ---
                if mano_nombre == "IZQUIERDA" and estado_orquesta == "PLAYING":
                    # Usamos el punto 8 (indice) - Invertimos Y (1-y) porque 0 es arriba
                    posI_y = 1.0 - current_smoothed[8][1]
                    
                    # Mapeo similar al InverseLerp (rango 0.2 a 0.8)
                    rango_min, rango_max = 0.2, 0.8
                    factorVol = (posI_y - rango_min) / (rango_max - rango_min)
                    factorVol = max(0.0, min(1.0, factorVol))
                    
                    volTarget = factorVol * factorVol # Curva cuadratica
                    
                    # Determinar si estamos subiendo o bajando antes de suavizar
                    direccion = "VOLUME_UP" if volTarget > volumenSuavizado else "VOLUME_DOWN"
                    
                    # Aplicar suavizado (Lerp)
                    volumenSuavizado += (volTarget - volumenSuavizado) * min(1.0, dt * inerciaVolumen)
                    
                    # Enviar a Unity: "VOL:0.75" y luego el nombre del gesto
                    send_gesture(f"VOL:{volumenSuavizado:.3f}")
                    # Opcional: enviar el nombre del gesto si tu Unity lo requiere especificamente
                    # send_gesture(direccion)

                # --- LoGICA DE ESTADOS (MUnhECA) ---
                muneca_y = current_smoothed[0][1]
                en_zona_media = False
                with m_altura_pecho_y, m_altura_cadera_y:
                    en_zona_media = altura_pecho_y < muneca_y < altura_cadera_y
                    bajo_la_cadera = muneca_y > altura_cadera_y

                with m_estado_orquesta:
                    if en_zona_media and estado_orquesta == "IDLE":
                        estado_orquesta = "READY"
                        send_gesture("READY")
                    elif bajo_la_cadera and estado_orquesta == "STOP":
                        estado_orquesta = "IDLE"
                        send_gesture("IDLE")

                # --- LoGICA DE DIRECCIoN (MANO DERECHA) ---
                if mano_nombre == "DERECHA":
                    dedo_y = current_smoothed[12][1]
                    historial_pos[h_idx].append((dedo_y, time.time()))
                    if len(historial_pos[h_idx]) > 8: historial_pos[h_idx].pop(0)

                    if len(historial_pos[h_idx]) >= 5:
                        subida = historial_pos[h_idx][0][0] - historial_pos[h_idx][-1][0]

                        with m_estado_orquesta:
                            if estado_orquesta == "READY" and subida > 0.06:
                                estado_orquesta = "PLAYING"
                                send_gesture("START")
                                historial_pos[h_idx] = []
                            elif estado_orquesta == "PLAYING":
                                x_pulgar = current_smoothed[4][0]
                                y_pulgar = current_smoothed[4][1]

                                x_indice = current_smoothed[8][0]
                                y_indice = current_smoothed[8][1]

                                d_pulgar_indice = math.sqrt(((x_pulgar - x_indice)**2) + ((y_pulgar - y_indice)**2))

                                if d_pulgar_indice < 0.03:
                                    estado_orquesta = "STOP"
                                    send_gesture("STOP")
                                    historial_pos[h_idx] = []

    # Configuracion de Mediapipe
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

        # Logica de lineas de pose para visualizacion
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
            cv2.putText(frame, f"VOL: {int(volumenSuavizado*100)}%", (10, 90), 2, 0.8, (0, 255, 0), 2)
            
        cv2.imshow("Director Console", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
