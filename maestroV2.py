import cv2
import mediapipe as mp
import math
import time
import socket
from multiprocessing import Lock
from enum import IntEnum

class MessageType(IntEnum):
    Ready = 0,
    Start = 1,
    Stop = 2,
    Calderon = 10,
    OffCalderon = 11,
    VolumeUp = 20,
    VolumeDown = 21,
    Volume = 22,
    Tempo = 30

# --- TCP/UDP Cliente ---
UDP = 0
TCP = 1

mode = UDP # o UDP

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM if mode == TCP else socket.SOCK_DGRAM)
SERVER_ADDR = ("localhost", 8090)

if mode == TCP:
    sock.connect(SERVER_ADDR)

def send_gesture(msg: bytes):
    if mode == TCP:
        sock.sendall(msg)
    else:
        sock.sendto(msg, SERVER_ADDR)
    print(f"TCP >> {str(msg)}")

# --- Variables Globales y Sincronizacion ---
m_estado_orquesta = Lock()
estado_orquesta = "IDLE"
tiempo_entrada_zona = 0

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

    historial_pos = {0: [], 1: []}
    m_historial_pos = Lock()

    UMBRAL_DIRECCION = 0.06
    DIR_ABJ_ARR = 0
    DIR_ARR_ABJ = 1

    last_dedos_estirados = {'value': False }

    def obtener_direccion_ver(historial: list[list[list[int]]]):
        if len(historial) < 8:
            return None
        inicio = historial[0][0][1]
        fin = historial[-1][0][1]
        diferencia = fin - inicio
        if abs(diferencia) < UMBRAL_DIRECCION:
            return None
        return DIR_ARR_ABJ if diferencia > 0 else DIR_ABJ_ARR


    def process_hands(result_hand, mp_image, timestamp_ms):
        global estado_orquesta, volumenSuavizado, last_time_vol, tiempo_entrada_zona

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
                if h_idx in prev_hands and len(prev_hands[h_idx]) == len(hand):
                    for lm_idx, lm in enumerate(hand):
                        prev_lm = prev_hands[h_idx][lm_idx]
                        x_s = ALPHA * lm.x + (1 - ALPHA) * prev_lm[0]
                        y_s = ALPHA * lm.y + (1 - ALPHA) * prev_lm[1]
                        z_s = ALPHA * lm.z + (1 - ALPHA) * prev_lm[2]
                        current_smoothed.append((x_s, y_s, z_s))
                else:
                    current_smoothed = [(lm.x, lm.y, lm.z) for lm in hand]

                prev_hands[h_idx] = current_smoothed

                # 2. Posicion de la muneca para estados
                muneca_y = current_smoothed[0][1]

                en_zona_media = False
                with m_limites:
                    en_zona_media = altura_pecho_y < muneca_y < altura_cadera_y

                # 3. Logica especifica por mano
                #if muneca_y > altura_cadera_y and estado_orquesta == "READY":
                #    estado_orquesta = "IDLE"
                if muneca_y > altura_cadera_y and estado_orquesta == "STOP":
                    estado_orquesta = "IDLE"

                elif en_zona_media and estado_orquesta == "IDLE":
                    if tiempo_entrada_zona == 0:
                        tiempo_entrada_zona = time.time()

                    elif en_zona_media and time.time() - tiempo_entrada_zona > 4:
                      estado_orquesta = "READY"
                      send_gesture(bytes([MessageType.Ready.value])) # Ready
                else:
                    tiempo_entrada_zona = 0

                # --- MANO IZQUIERDA: CONTROL DE VOLUMEN ---
                if mano_nombre == "IZQUIERDA" and estado_orquesta == "PLAYING":

                    es_palma = current_smoothed[4][0] > current_smoothed[20][0]

                    dedos_estirados = (
                        current_smoothed[8][1] < current_smoothed[6][1] and
                        current_smoothed[12][1] < current_smoothed[10][1] and
                        current_smoothed[16][1] < current_smoothed[14][1] and
                        current_smoothed[20][1] < current_smoothed[18][1]
                    )

                    historial_pos[h_idx].append((current_smoothed[4], time.time()))

                    if len(historial_pos[h_idx]) > 30:
                        historial_pos[h_idx].pop(0)

                    # Resetear historial al detectar un cambio de dedos estirados
                    if dedos_estirados != last_dedos_estirados["value"]:
                        historial_pos[h_idx] = []

                    # print(last_dedos_estirados["value"], dedos_estirados)
                    last_dedos_estirados["value"] = dedos_estirados

                    # Ver si hay al menos 5 muestras que coincidan con dedos estirados/sin estirar
                    if len(historial_pos[h_idx]) >= 5:
                        # direccion_hor = obtener_direccion_hor(historial_pos[h_idx])
                        direccion_ver = obtener_direccion_ver(historial_pos[h_idx])

                    # Detección gestos
                        if dedos_estirados:

                            if not es_palma:

                                if direccion_ver == DIR_ABJ_ARR:
                                    posI_y = 1.0 - current_smoothed[8][1] # Invertir eje Y
                                    factorVol = max(0.0, min(1.0, (posI_y - 0.2) / 0.6))
                                    volTarget = factorVol ** 2
                                    volumenSuavizado += (volTarget - volumenSuavizado) * min(1.0, dt * inerciaVolumen)
                                    send_gesture(bytes([MessageType.Volume.value, round(volumenSuavizado * 100)]))

                        if es_palma:
                            if direccion_ver == DIR_ARR_ABJ:
                                posI_y = 1.0 - current_smoothed[8][1] # Invertir eje Y
                                factorVol = max(0.0, min(1.0, (posI_y - 0.2) / 0.6))
                                volTarget = factorVol ** 2
                                volumenSuavizado += (volTarget - volumenSuavizado) * min(1.0, dt * inerciaVolumen)
                                send_gesture(bytes([MessageType.Volume.value, round(volumenSuavizado * 100)]))

                # --- MANO DERECHA: START Y STOP ---
                if mano_nombre == "IZQUIERDA":
                    dedo_y = current_smoothed[12][1]
                    historial_pos[h_idx].append(((0, dedo_y, 0), time.time()))

                    # Reducimos el historial a 8 para más velocidad de respuesta
                    if len(historial_pos[h_idx]) > 8: historial_pos[h_idx].pop(0)

                    if len(historial_pos[h_idx]) >= 5:
                        # Calculamos la subida (valor inicial Y - valor final Y)
                        # En MediaPipe, subir es que Y disminuya, por eso inicial - final
                        subida = historial_pos[h_idx][0][0][1] - historial_pos[h_idx][-1][0][1]

                        # START: Sensibilidad alta (0.06 es suficiente para un latigazo)
                        with m_estado_orquesta:
                            if estado_orquesta == "READY" and subida > 0.26:
                                estado_orquesta = "PLAYING"
                                send_gesture(bytes([MessageType.Start.value])) # Start
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
                                    send_gesture(bytes([MessageType.Stop.value])) # Stop
                                    historial_pos[h_idx] = [] # Limpiar para evitar doble disparo

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
            y_cadera = ((lm[23].y + lm[24].y) / 2)


            if estado_orquesta != "PLAYING" and estado_orquesta != "STOP":
                 with m_limites:
                    global altura_pecho_y, altura_cadera_y
                    # Calculo de lineas de referencia
                    altura_pecho_y = y_hombros + (y_cadera - y_hombros) * 0.35
                    altura_cadera_y = y_cadera - 0.1
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
