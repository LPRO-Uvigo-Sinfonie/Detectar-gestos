import cv2
import mediapipe as mp
import math
import time
import socket
from multiprocessing import Lock

# --- Configuracion TCP ---
TCP_IP = "127.0.0.1"
TCP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((TCP_IP, TCP_PORT))
    print("Conectado a Unity con exito.")
except Exception as e:
    print(f"Error conectando a Unity: {e}")

def send_gesture(msg):
    try:
        sock.sendall(msg.encode())
        print(f"TCP >> {msg}")
    except:
        pass

# --- Variables Globales y Sincronizacion ---
m_estado_orquesta = Lock()
estado_orquesta = "IDLE"

# Volumen (Mano Izquierda)
volumenSuavizado = 0.5
last_time_vol = time.time()
inerciaVolumen = 8.0

# Limites de Pose
m_limites = Lock()
altura_pecho_y = 0.4
altura_cadera_y = 0.8

def main():
    model_path_hand = "hand_landmarker.task"
    model_path_pose = "pose_landmarker.task"

    # Parametros de suavizado y deteccion
    ALPHA = 0.65
    prev_hands = {}
    historial_pos_derecha = [] # Para el latigazo de START

    def process_hands(result_hand, mp_image, timestamp_ms):
        global estado_orquesta, volumenSuavizado, last_time_vol
        
        current_time = time.time()
        dt = current_time - last_time_vol
        last_time_vol = current_time

        if result_hand.hand_landmarks:
            manos_en_zona_media = 0
            manos_bajo_cadera = 0
            
            for h_idx, hand in enumerate(result_hand.hand_landmarks):
                # En modo espejo (flip 1), "Left" es la mano derecha del usuario
                label = result_hand.handedness[h_idx][0].category_name
                mano_nombre = "DERECHA" if label == "Left" else "IZQUIERDA"

                # 1. Suavizado de puntos (EMA)
                current_smoothed = []
                if h_idx in prev_hands:
                    for i, lm in enumerate(hand):
                        prev = prev_hands[h_idx][i]
                        current_smoothed.append((ALPHA*lm.x + (1-ALPHA)*prev[0], ALPHA*lm.y + (1-ALPHA)*prev[1]))
                else:
                    current_smoothed = [(lm.x, lm.y) for lm in hand]
                prev_hands[h_idx] = current_smoothed

                # 2. Posicion de la muneca para estados
                muneca_y = current_smoothed[0][1]
                with m_limites:
                    en_zona = altura_pecho_y < muneca_y < altura_cadera_y
                    bajo_cadera = muneca_y >= altura_cadera_y
                
                if en_zona: manos_en_zona_media += 1
                if bajo_cadera: manos_bajo_cadera += 1

                # 3. Logica especifica por mano
                with m_estado_orquesta:
                    # --- MANO IZQUIERDA: CONTROL DE VOLUMEN ---
                    if mano_nombre == "IZQUIERDA" and estado_orquesta == "PLAYING":
                        posI_y = 1.0 - current_smoothed[8][1] # Invertir eje Y
                        factorVol = max(0.0, min(1.0, (posI_y - 0.2) / 0.6))
                        volTarget = factorVol ** 2
                        volumenSuavizado += (volTarget - volumenSuavizado) * min(1.0, dt * inerciaVolumen)
                        send_gesture(f"VOL:{volumenSuavizado:.3f}")

                    # --- MANO DERECHA: START Y STOP ---
                    if mano_nombre == "DERECHA":
                        # Deteccion de START (Latigazo hacia arriba)
                        dedo_y = current_smoothed[12][1]
                        historial_pos_derecha.append(dedo_y)
                        if len(historial_pos_derecha) > 8: historial_pos_derecha.pop(0)

                        if estado_orquesta == "READY" and len(historial_pos_derecha) >= 5:
                            subida = historial_pos_derecha[0] - historial_pos_derecha[-1]
                            if subida > 0.06:
                                estado_orquesta = "PLAYING"
                                send_gesture("START")
                                historial_pos_derecha.clear()

                        # Deteccion de STOP (Gesto de pinza)
                        elif estado_orquesta == "PLAYING":
                            # Distancia entre punta pulgar (4) e indice (8)
                            dist = math.sqrt((current_smoothed[4][0] - current_smoothed[8][0])**2 + 
                                             (current_smoothed[4][1] - current_smoothed[8][1])**2)
                            if dist < 0.03:
                                estado_orquesta = "STOP"
                                send_gesture("STOP")

            # 4. Transiciones globales de estado (READY / IDLE)
            with m_estado_orquesta:
                # Si estamos en IDLE y subimos al menos una mano al medio
                if estado_orquesta == "IDLE" and manos_en_zona_media >= 1:
                    estado_orquesta = "READY"
                    send_gesture("READY")
                
                # Si bajamos las manos estando en READY o tras un STOP
                elif (estado_orquesta == "READY" or estado_orquesta == "STOP") and manos_bajo_cadera >= 1:
                    estado_orquesta = "IDLE"
                    send_gesture("IDLE")

    # --- Configuracion de Tareas de Mediapipe ---
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options_hand = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_hand),
        running_mode=VisionRunningMode.LIVE_STREAM, 
        num_hands=2, 
        result_callback=process_hands)
    
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

        # 1. Deteccion de Pose (para limites dinamicos)
        result_pose = detector_pose.detect_for_video(mp_image, timestamp_ms)
        if result_pose.pose_landmarks:
            lm = result_pose.pose_landmarks[0]
            y_hombros = (lm[11].y + lm[12].y) / 2
            y_cadera = (lm[23].y + lm[24].y) / 2
            
            with m_limites:
                global altura_pecho_y, altura_cadera_y
                # Calculo de lineas de referencia
                altura_pecho_y = y_hombros + (y_cadera - y_hombros) * 0.35
                altura_cadera_y = y_cadera
                # Dibujar guias visuales
                cv2.line(frame, (0, int(altura_pecho_y*h_img)), (w_img, int(altura_pecho_y*h_img)), (0, 255, 255), 2)
                cv2.line(frame, (0, int(altura_cadera_y*h_img)), (w_img, int(altura_cadera_y*h_img)), (0, 0, 255), 2)

        # 2. Deteccion de Manos (Asincrona)
        detector_hand.detect_async(mp_image, timestamp_ms)

        # 3. Interfaz de Usuario
        with m_estado_orquesta:
            cv2.putText(frame, f"ESTADO: {estado_orquesta}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if estado_orquesta == "PLAYING":
                cv2.putText(frame, f"VOL: {int(volumenSuavizado*100)}%", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Director Console", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()