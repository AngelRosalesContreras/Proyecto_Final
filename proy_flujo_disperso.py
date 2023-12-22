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

def apply_kmeans_to_freeman_list(img, contours, freeman_chaincodes,roi_x, roi_y):
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
    optimal_k = elbow_method(features_array, max_clusters=10)
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
    for i, contour in enumerate(contours):
        cluster_color = get_random_color(labels[i])
        cv2.drawContours(clustered_frame, [contour], -1, (0, 255, 0), 2, offset=(roi_x, roi_y))
        cv2.drawContours(clustered_frame, [contour], -1, cluster_color, -1, offset=(roi_x, roi_y))


    img_resized = cv2.resize(clustered_frame, (width, height))
    cv2.imshow("Clustered Objects",img_resized )
    cv2.waitKey(0)  # Esperar hasta que se presione una tecla para cerrar la ventana

    return labels

def initialize_params():
    lk_params = dict(winSize=(30, 30),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))

    feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=10,
                      blockSize=7)
    return lk_params, feature_params

def process_roi(frame, trajectories, min_contour_area, max_contour_area, contours_dict, img):
    if len(trajectories) > 0:
        all_points = np.concatenate([np.int32(trajectory) for trajectory in trajectories])
        x, y, w, h = cv2.boundingRect(all_points)

        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and h > 0 and w > 0:
            roi = frame[y:y + h, x:x + w]
            roi_x, roi_y = x, y  # Agrega estas líneas

            if roi.size != 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(roi_gray, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contourss, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filtered_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

                for trajectory in trajectories:
                    for x, y in np.int32(trajectory):
                        # Dibujar punto
                        cv2.circle(img, (x, y), 1, (255, 0, 255), -1)

                for contour in filtered_contours:
                    # No agregar x e y aquí
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x + roi_x, y + roi_y), (x + w + roi_x, y + h + roi_y), (255, 0, 0), 2)

                    contour_id = id(contour)

                    if contour_id not in contours_dict:
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        contours_dict[contour_id] = color
                    else:
                        color = contours_dict[contour_id]

                    # No agregar x e y aquí
                    cv2.drawContours(img, [contour + (roi_x, roi_y)], -1, color, 2)

                all_freeman_chaincodes = extract_freeman_from_contours(contourss)
                kmeans_labels = apply_kmeans_to_freeman_list(frame, contourss, all_freeman_chaincodes,roi_x, roi_y)
                print("K-Means Labels:", kmeans_labels)

def detect_features(frame_gray, trajectories, lk_params,img):
    img0, img1 = prev_gray, frame_gray
    p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    good = d < 1

    new_trajectories = []

    # Obtener todas las trayectorias
    for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
        if not good_flag:
            continue
        trajectory.append((x, y))
        if len(trajectory) > trajectory_len:
            del trajectory[0]
        new_trajectories.append(trajectory)
        # Punto más recientemente detectado
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    trajectories = new_trajectories

    return  trajectories

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


trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

video_path = 'tres.mp4'
cap = cv2.VideoCapture(video_path)
# Después de abrir el video, antes de entrar al bucle while
width = 1366  # El ancho deseado
height = 768  # La altura deseada
# width = 600  # El ancho deseado
# height = 800  # La altura deseada

# Asegúrate de que el video tenga el tamaño deseado
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Diccionario para realizar un seguimiento de los contornos y sus colores
contours_dict = {}

# Umbral inferior y superior para el área del contorno
min_contour_area = 600
max_contour_area = 7000

while True:
    # Tiempo de inicio para calcular los FPS
    start = time.time()
    suc, frame = cap.read()

    if not suc or frame is None:
        break  # Si no se puede leer el siguiente cuadro, sal del bucle

    lk_params, feature_params = initialize_params()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    #  Método  Lucas-Kanade
    if len(trajectories) > 0:
        trajectories = detect_features(frame_gray, trajectories, lk_params, img)
        process_roi(frame, trajectories, min_contour_area, max_contour_area, contours_dict, img)

    # Intervalo de actualización: cuando actualizar y detectar nuevas características
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Último punto en la trayectoria más reciente
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detectar las buenas características para seguir
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            # Si se pueden seguir buenas características, agregarlas a las trayectorias
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # Calcular los FPS (cuadros por segundo) para la detección del cuadro actual
    fps = 1 / (end - start)

    # Mostrar resultados
    # Redimensiona la imagen para mostrarla con el tamaño deseado
    img_resized = cv2.resize(img, (width, height))

    # cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Optical Flow', img_resized)
    # cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
