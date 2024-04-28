import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib.animation import FuncAnimation

# Générer des données circulaires non linéairement séparables en 2D
np.random.seed(0)
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1)

# Ajouter une colonne de zéros à la matrice X pour obtenir une représentation en 3D
X = np.hstack([X, np.zeros((X.shape[0], 1))])

# Initialiser le modèle SVM avec un noyau gaussien
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1))
svm.fit(X[:, :2], y)  # Nous utilisons uniquement les deux premières colonnes de X pour l'entraînement

# Transformer les données en 3D avec le kernel trick
X_3d = svm.decision_function(X[:, :2]).reshape(-1, 1)  # Mettre à jour la colonne z avec les valeurs du kernel trick

# Créer une figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer les points de données 2D initiaux
scatter = ax.scatter(X[:, 0], X[:, 1], np.zeros_like(X[:, 0]), c=y, cmap=plt.cm.coolwarm)

# Définir les limites des axes pour une meilleure visualisation
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])  # Pour inclure la variation sur l'axe z
ax.view_init(azim=-60, elev=20)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('')

# Calculer les positions interpolées des points de données
def calculate_interpolated_positions(X, X_3d, frame):
    interpolated_positions = X + (X_3d - X) * (frame / 100)
    return interpolated_positions

# Fonction pour mettre à jour l'animation
def update(frame):
    interpolated_positions = calculate_interpolated_positions(X, X_3d, frame)  # Nous utilisons X_3d pour les positions finales
    # Mettre à jour les positions des points de données
    scatter._offsets3d = (interpolated_positions[:, 0], interpolated_positions[:, 1], interpolated_positions[:, 2])
    
    # Ajouter un plan séparateur dans les dernières frames de l'animation
    if frame >= 85:
        # Séparer les données par classe
        class_1_indices = np.where(y == 0)[0]
        class_2_indices = np.where(y == 1)[0]
        # Choisir un point dans chaque classe
        point_1 = interpolated_positions[class_1_indices[0]]
        point_2 = interpolated_positions[class_2_indices[0]]
        # Calculer le vecteur normal au plan séparateur
        normal_vector = point_2 - point_1
        # Normaliser le vecteur
        normal_vector /= np.linalg.norm(normal_vector)
        # Définir le plan séparateur à partir du vecteur normal
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
        zz = (-normal_vector[0] * xx - normal_vector[1] * yy) / normal_vector[2]
        # Tracer le plan séparateur
        ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5,label="Hyperplane")
    
    return scatter,

# Créer l'animation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 100, 100), interval=30)
plt.show()
