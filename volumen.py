import cv2
import mediapipe as mp
from numba import njit
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import os
from typing import Literal, TypedDict
import socket

class TManosProcesadas(TypedDict):
    landmarks: list[tuple[int, int, int]]
    nombre: Literal[0, 1]
    posicion: Literal["ARRIBA", "ABAJO", ""]
    es_palma: bool
    color: tuple[int, int, int]

def main(): 

    # --- UDP Cliente ---

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SERVER_ADDR = ("127.0.0.1", 5005)
    client_socket.connect(SERVER_ADDR)

    # --- MODELOS ---
    model_path_hand = "hand_landmarker.task"
    model_path_pose = "pose_landmarker.task"
    
    if not os.path.exists(model_path_hand) or not os.path.exists(model_path_pose):
        print("Faltan modelos .task")
        exit(1)
    
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), 
                        (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), 
                        (0,17), (17,18), (18,19), (19,20)]
    
    POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    
    DIR_IZQ_DER = 0
    DIR_DER_IZQ = 1

    DIR_ABJ_ARR = 0
    DIR_ARR_ABJ = 1

    MANO_IZQ = 0
    MANO_DER = 1

    CODO_IZQ = 14
    CODO_DER = 13

    alpha = 0.65
    prev_hands = {}
    historial_pos = {0: [], 1: []}
    last_gestos = {
        0: {'cena': 0, 'saludo': 0, 'dias': 0, 'noches': 0, 'subir': 0, 'bajar': 0},
        1: {'cena': 0, 'saludo': 0, 'dias': 0, 'noches': 0, 'subir': 0, 'bajar': 0}
    }
    
    UMBRAL_DIRECCION = 0.06
    PIXELES_REF_CERCA = 350

    def process_hands(result_hand: HandLandmarkerResult, mp_image: image_lib.Image, timestamp_ms: int):
        hands_landmarks = result_hand.hand_landmarks
        handedness = result_hand.handedness

        linea_codos_y = None
        # if result_pose.pose_landmarks:
        #     linea_codos_y = calcular_linea_codos(result_pose.pose_landmarks[0], h_img)
        
        manos_procesadas: list[TManosProcesadas] = []
        mensajes: list[str] = []
    
        for h_idx, hand in enumerate(hands_landmarks):
            label = handedness[h_idx][0].category_name
            mano_nombre = MANO_DER if label == "Left" else MANO_IZQ # Esta linea tiene sentido
            
            current_smoothed = []
            if h_idx in prev_hands and len(prev_hands[h_idx]) == len(hand):
                for lm_idx, lm in enumerate(hand):
                    prev_lm = prev_hands[h_idx][lm_idx]
                    x_s = alpha * lm.x + (1 - alpha) * prev_lm[0]
                    y_s = alpha * lm.y + (1 - alpha) * prev_lm[1]
                    z_s = alpha * lm.z + (1 - alpha) * prev_lm[2]
                    current_smoothed.append((x_s, y_s, z_s))
            else:
                current_smoothed = [(lm.x, lm.y, lm.z) for lm in hand]
            
            prev_hands[h_idx] = current_smoothed
            
            posicion = None
            if linea_codos_y:
                muneca_y = int(current_smoothed[0][1] * h_img)
                posicion = "ARRIBA" if muneca_y < linea_codos_y else "ABAJO"
            
            if mano_nombre == MANO_DER:
                es_palma = current_smoothed[4][0] < current_smoothed[20][0]
            else:
                es_palma = current_smoothed[4][0] > current_smoothed[20][0]
            
            dedos_estirados = (
                current_smoothed[8][1] < current_smoothed[6][1] and
                current_smoothed[12][1] < current_smoothed[10][1] and
                current_smoothed[16][1] < current_smoothed[14][1] and
                current_smoothed[20][1] < current_smoothed[18][1]
            )
            
            mensaje = None
            
            if dedos_estirados:
                historial_pos[h_idx].append((current_smoothed[4], time.time()))

                if len(historial_pos[h_idx]) > 15:
                    historial_pos[h_idx].pop(0)
            else:
                historial_pos[h_idx] = []
                
            if len(historial_pos[h_idx]) >= 10:
                direccion_hor = obtener_direccion_hor(historial_pos[h_idx])
                direccion_ver = obtener_direccion_ver(historial_pos[h_idx])
                ahora = time.time()

            # Detección gestos
                if dedos_estirados:
                        
                    if not es_palma:

                        if direccion_ver == DIR_ABJ_ARR and ahora - last_gestos[h_idx]['subir'] >= 1 and mano_nombre == MANO_IZQ:
                            client_socket.sendall(b"VOLUME_UP")
                            mensaje = f"Subir volumen"
                            last_gestos[h_idx]['subir'] = ahora
                            historial_pos[h_idx] = []

                if es_palma:
                    if direccion_ver == DIR_ARR_ABJ and ahora - last_gestos[h_idx]['bajar'] >= 1 and mano_nombre == MANO_IZQ:
                        client_socket.sendall(b"VOLUME_DOWN")
                        mensaje = f"Bajar volumen"
                        last_gestos[h_idx]['bajar'] = ahora
                        historial_pos[h_idx] = []
                
            if mensaje:
                mensajes.append(mensaje)
            
            manos_procesadas.append({
                'landmarks': current_smoothed,
                'nombre': mano_nombre,
                'posicion': posicion,
                'es_palma': es_palma,
                'color': (0, 255, 255) if mano_nombre == MANO_DER else (255, 0, 255)
            })

    options_hand = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_hand),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.2,
        result_callback=process_hands
    )
    
    options_pose = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_pose),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )
    
    detector_hand = mp.tasks.vision.HandLandmarker.create_from_options(options_hand)
    detector_pose = mp.tasks.vision.PoseLandmarker.create_from_options(options_pose)
    
    cap = cv2.VideoCapture(0)
    
    print(f"Resolución actual: {int(cap.get(3))}x{int(cap.get(4))}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Resolución final: {int(cap.get(3))}x{int(cap.get(4))} @ {int(cap.get(5))}FPS")
    
    @njit
    def pixeles_a_cm(pixeles: int):
        if pixeles <= 0:
            return 0
        return min(max(int((PIXELES_REF_CERCA * 20) / pixeles), 10), 150)
    
    @njit
    def obtener_direccion_hor(historial: list[list[list[int]]]):
        if len(historial) < 10:
            return None
        inicio = historial[0][0][0]
        fin = historial[-1][0][0]
        diferencia = fin - inicio
        if abs(diferencia) < UMBRAL_DIRECCION:
            return None
        return DIR_IZQ_DER if diferencia > 0 else DIR_DER_IZQ
    
    @njit
    def obtener_direccion_ver(historial: list[list[list[int]]]):
        if len(historial) < 10:
            return None
        inicio = historial[0][0][1]
        fin = historial[-1][0][1]
        diferencia = fin - inicio
        if abs(diferencia) < UMBRAL_DIRECCION:
            return None
        return DIR_ARR_ABJ if diferencia > 0 else DIR_ABJ_ARR

    def calcular_linea_codos(pose_landmarks: list[landmark_lib.NormalizedLandmark], h_img: int):
        if len(pose_landmarks) < 13:
            return None

        y_izq = int(pose_landmarks[CODO_DER].y * h_img)
        y_der = int(pose_landmarks[CODO_IZQ].y * h_img)
        return (y_izq + y_der) // 2
    
    tiempo_anterior = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        tiempo_actual = time.time()
        fps = 1 / (tiempo_actual - tiempo_anterior + 0.001)
        tiempo_anterior = tiempo_actual
        
        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        detector_hand.detect_async(mp_image, timestamp_ms)
        result_pose = detector_pose.detect_for_video(mp_image, timestamp_ms)
    
        cv2.imshow("Detector i7 - Manos + Codos + Linea Codos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

main()