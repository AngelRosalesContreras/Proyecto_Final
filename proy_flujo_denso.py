import numpy as np
import cv2
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Diccionario para almacenar colores únicos por clúster
cluster_colors = {}

def get_random_color(cluster):
    # Verificar si ya se asignó un color para este clúster
    if cluster not in cluster_colors:
        # Si no se ha asignado, generar un color aleatorio
        cluster_colors[cluster] = tuple(map(int, np.random.randint(0, 256, 3)))
    return cluster_colors[cluster]

def elbow_method(data, max_clusters=10):
    distortions = []
    n_samples = len(data)

    for i in range(1, min(max_clusters, n_samples) + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    # Calcular la derivada segunda de la curva de distorsión
    acceleration = np.diff(np.diff(distortions))

    # Encontrar el índice donde la aceleración es máxima
    optimal_k_index = np.argmax(acceleration) + 2  # Sumar 2 para compensar la doble diferenciación

    # Graficar la curva de distorsión y resaltar el punto óptimo
    plt.plot(range(1, min(max_clusters, n_samples) + 1), distortions, marker='o')
    plt.scatter(optimal_k_index, distortions[optimal_k_index - 1], c='red', label='Optimal k')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.legend()
    plt.show()

    # Devolver el número óptimo de clústeres
    return optimal_k_index


def apply_kmeans_to_freeman_list(img,contours,freeman_chaincodes):
    if not freeman_chaincodes:
        print("No hay datos para aplicar K-Means.")
        return []

    # Obtener características de las cadenas de Freeman
    def get_freeman_features(chain_code):
        length = len(chain_code)
        direction_changes = sum(1 for a, b in zip(chain_code, chain_code[1:]) if a != b)
        return length, direction_changes

    # Obtener características para cada cadena de Freeman
    features = [get_freeman_features(chain_code) for chain_code in freeman_chaincodes]
    features_array = np.array(features)

    if features_array.ndim == 1:
        # Si es un array 1D, redimensiona a 2D
        features_array = features_array.reshape(-1, 1)

    # Aplicar el método del codo para determinar el número óptimo de clústeres
    optimal_k = elbow_method(features_array,max_clusters=10)
    print(optimal_k)

    # Aplicar k-medias con el número óptimo de clústeres
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(features_array)

    # Obtener etiquetas para cada cadena de Freeman
    labels = kmeans.labels_

    # Visualizar en un gráfico de dispersión
    lengths, changes = zip(*[get_freeman_features(chain_code) for chain_code in freeman_chaincodes])
    plt.scatter(lengths, changes, c=labels, cmap='viridis', marker='o', edgecolors='k', label='Data Points')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200,
                label='Centroids')
    plt.title('K-Means Clustering of Freeman Chain Codes with Centroids')
    plt.xlabel('Length of Chain Code')
    plt.ylabel('Number of Direction Changes')
    plt.legend()
    plt.show()

    # Crear una imagen para visualizar elementos de cada clúster en el frame
    clustered_frame = img.copy()

    # Encontrar contornos en la máscara
    # contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cluster_color = get_random_color(labels[i])
        cv2.drawContours(clustered_frame, [contour], -1, (0, 255, 0), 2)
        # Asignar un color específico para cada clúster
        # color = tuple(map(int, plt.cm.viridis(labels[i] / optimal_k)[:3] * 255))
        cv2.drawContours(clustered_frame, [contour], -1,  cluster_color, -1)  # Rellenar el contorno con el color del clúster

    cv2.imshow("Clustered Objects", clustered_frame)
    cv2.waitKey(0)  # Esperar hasta que se presione una tecla para cerrar la ventana

    return labels



def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang * (180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def get_freeman_chaincode(contour):
    chain_code = []
    direction_map = [0, 1, 2, 3, 4, 5, 6, 7]

    for i in range(1, len(contour)):
        delta_x = contour[i][0][0] - contour[i - 1][0][0]
        delta_y = contour[i][0][1] - contour[i - 1][0][1]

        # Determinar la dirección del movimiento
        direction = (delta_x > 0) + (delta_y > 0) * 2 + (delta_x < 0) * 4 + (delta_y < 0) * 6

        try:
            chain_code.append(direction_map.index(direction))
        except ValueError:
            # Si la dirección no está en la lista, manejar la excepción
            pass

    return chain_code

def extract_freeman_from_contours(contours):
    freeman_chaincodes = []

    for contour in contours:
        freeman_chaincode = get_freeman_chaincode(contour)
        freeman_chaincodes.append(freeman_chaincode)

    return freeman_chaincodes

def draw_contours_on_image(image, contours):
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    return image_with_contours

def segment_objects_with_optical_flow(frame, flow, threshold=1.0):
    # Obtener la magnitud del flujo óptico
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Aplicar umbralización para resaltar las áreas de cambio
    motion_mask = (magnitude > threshold).astype(np.uint8) * 255

    # Aplicar filtros morfológicos para mejorar la máscara
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    # Crear una imagen en blanco
    blue_background = np.zeros_like(frame)
    # Asignar color azul a las áreas de cambio
    blue_background[motion_mask > 0] = (255, 0, 0)

    # Combinar la imagen original con el fondo azul
    segmented_objects = cv2.addWeighted(frame, 1, blue_background, 0.5, 0)

    # Crear una imagen en blanco para la ROI
    frame_with_freeman = frame.copy()

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la ROI
    # Listas para almacenar las cadenas de Freeman
    all_freeman_chaincodes = []

    for contour in contours:
        cv2.drawContours(frame_with_freeman, [contour], -1, (0, 255, 0), 2)

        # Obtener la cadena de contornos de Freeman
        freeman_chaincode = get_freeman_chaincode(contour)

        cv2.imshow("Freeman Contour", frame_with_freeman)
        all_freeman_chaincodes.append(freeman_chaincode)

        # print("Freeman Chain Code:", freeman_chaincode)
    print("Freeman Chain Code:", all_freeman_chaincodes)
    kmeans_labels = apply_kmeans_to_freeman_list(frame,contours, all_freeman_chaincodes)
    print("K-Means Labels:", kmeans_labels)

    return segmented_objects

# Código principal aquí
video_path = 'nofi001.mp4'
cap = cv2.VideoCapture(video_path)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    suc, img = cap.read()
    if not suc:
        break  # Si no se puede leer el siguiente cuadro, sal del bucle

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Iniciar el tiempo para calcular el FPS
    start = time.time()

    # Calcular el flujo óptico
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    prevgray = gray

    end = time.time()
    # Calcular el FPS para la detección del cuadro actual
    fps = 1 / (end - start)

    print(f"{fps:.2f} FPS")

    cv2.imshow("Flow", draw_flow(gray, flow))
    cv2.imshow("HSV", draw_hsv(flow))

    # Segmentar objetos en movimiento y marcarlos en color azul
    segmented_objects = segment_objects_with_optical_flow(img, flow)

    cv2.imshow("Segmented Objects", segmented_objects)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    # Agregar una pausa para limitar la velocidad de procesamiento a 2 FPS
    time.sleep(1)
cap.release()
cv2.destroyAllWindows()