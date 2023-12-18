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


def apply_kmeans_to_trajectory_list(trajectories, k=3):
    if not trajectories:
        print("No hay datos para aplicar K-Means.")
        return []

    # Obtener características de las trayectorias
    def get_trajectory_features(trajectory):
        length = len(trajectory)
        return length

    # Obtener características para cada trayectoria
    features = [get_trajectory_features(trajectory) for trajectory in trajectories]
    features_array = np.array(features).reshape(-1, 1)

    # Aplicar K-medias con el método del codo
    optimal_k = elbow_method(features_array, max_clusters=10)
    print(optimal_k)
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(features_array)

    # Asignar etiquetas a cada trayectoria
    labels = kmeans.labels_

    # Visualizar en un gráfico de dispersión
    lengths = [get_trajectory_features(trajectory) for trajectory in trajectories]
    plt.scatter(range(len(trajectories)), lengths, c=labels, cmap='viridis', marker='o', edgecolors='k', label='Data Points')
    plt.scatter(range(optimal_k), [kmeans.cluster_centers_[i, 0] for i in range(optimal_k)],
                c='red', marker='x', s=200, label='Centroids')
    plt.title('K-Means Clustering of Trajectories with Centroids')
    plt.xlabel('Trajectory Index')
    plt.ylabel('Length of Trajectory')
    plt.legend()
    plt.show()

    return labels


# Función para dibujar trayectorias y puntos más recientes
def draw_trajectories(img, trajectories, labels):
    clustered_frame = img.copy()

    for i, (trajectory, label) in enumerate(zip(trajectories, labels)):
        cv2.polylines(img, [np.int32(trajectory)], False, (0, 255, 0))
        x, y = np.int32(trajectory[-1])
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        cluster_color = get_random_color(label)
        cv2.polylines(clustered_frame, [np.int32(trajectory)], False, cluster_color)
        cv2.circle(clustered_frame, (x, y), 2, cluster_color, -1)

    cv2.imshow('Clustered Objects', clustered_frame)
    cv2.waitKey(0)  # Esperar hasta que se presione una tecla para cerrar la ventana


# Método del codo para determinar el número óptimo de clústeres
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


# Código principal
video_path = 'uno.mp4'
cap = cv2.VideoCapture(video_path)

lk_params = dict(winSize=(30, 30),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

while True:
    start = time.time()

    suc, frame = cap.read()

    if not suc or frame is None:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        good = np.all(np.ones_like(p1).astype(bool), axis=-1)

        new_trajectories = []

        for trajectory, (x, y), is_good in zip(trajectories, p1.reshape(-1, 2), good):
            if not is_good:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)

        trajectories = new_trajectories

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    # Aplicar K-medias y obtener etiquetas
    kmeans_labels = apply_kmeans_to_trajectory_list(trajectories, k=2)
    print("K-Means Labels:", kmeans_labels)

    # Dibujar trayectorias y puntos más recientes
    draw_trajectories(frame, trajectories, kmeans_labels)

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(frame, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Optical Flow', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
