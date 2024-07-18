import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib.colors import ListedColormap

# Générer des données circulaires non linéairement séparables en 2D

np.random.seed(0)
X, y = make_circles(n_samples=100, factor=0.25, noise=0.1)

# Ajouter une colonne de zéros à la matrice X pour obtenir une représentation en 3D
X = np.hstack([X, np.zeros((X.shape[0], 1))])


# Initialiser le modèle SVM avec un noyau gaussien
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf',degree=2, C=1))
svm.fit(X[:, :2], y)  # Nous utilisons uniquement les deux premières colonnes de X pour l'entraînement

# Transformer les données en 3D avec le kernel trick
X_3d = svm.decision_function(X[:, :2]).reshape(-1, 1)  # Mettre à jour la colonne z avec les valeurs du kernel trick

# Créer une figure
fig = plt.figure(dpi=300,layout='tight')
#fig.set_size_inches(6, 6)
#fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.tight_layout()

# Tracer les points de données 2D initiaux
cmap0 = sns.cubehelix_palette(n_colors=5, as_cmap=True)
colors =cmap0(np.linspace(0, 1, 5))
cmap = ListedColormap(colors[2:4])
scatter = ax.scatter(X[:, 0], X[:, 1], np.zeros_like(X[:, 0]), c=y, cmap=cmap, alpha=0.8, s=50)

# Définir les limites des axes pour une meilleure visualisation
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])  # Pour inclure la variation sur l'axe z
#ax.view_init(azim=-60, elev=20) #90;-90
ax.view_init(azim=-90, elev=90)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


#plt.show()
ax.axis('off')
# Tracer le plan z=0 en gris
x0, y0 = np.meshgrid(np.linspace(-2, 2, 2), np.linspace(-2, 2, 2))
z0 = np.zeros_like(x0)
ax.plane_surface = ax.plot_surface(x0, y0, z0, color='gray', alpha=0.10)
ax.hyperplan = None
# Calculer les positions interpolées des points de données
def calculate_interpolated_positions(X, X_3d, frame):
    interpolated_positions = X + (X_3d - X) * (frame / 100)
    return interpolated_positions

# Calculer l'opacité en fonction de la frame
def calculate_opacity(frame):
    if frame <= -2:
        return -frame * 0.03  # Réduire l'opacité de 0.5 à 0.0 sur les 10 premières frames
    else:
        return 0
    
view_start = {'azim': -90, 'elev': 90}
view_end = {'azim': -60, 'elev': 20}


def interpolate_view(view1, view2, t):
    azim = view1['azim'] * (1 - np.sin(t * np.pi / 2)) + view2['azim'] * np.sin(t * np.pi / 2)
    elev = view1['elev'] * (1 - t) + view2['elev'] * t
    return {'azim': azim, 'elev': elev}

def change_view(view):
    ax.view_init(elev=view['elev'], azim=view['azim'])

def set_axis_properties(axis, alpha):
    for spine in ['xaxis', 'yaxis', 'zaxis']:
        axis = eval(f'ax.{spine}')
        axis.line.set_alpha(alpha)
        for line in axis.get_gridlines():
            line.set_alpha(alpha)
        for line in axis.get_ticklines():
            line.set_alpha(alpha)
        for label in axis.get_ticklabels():
            label.set_alpha(alpha)
        axis.pane.set_edgecolor((1, 1, 1, alpha))  # Set pane edge color with alpha
        #axis.pane.set_facecolor((0, 0, 0, 0))     # Set pane face color to transparent

frame_list = np.linspace(-100, 100, 130)#np.arange(-71,100)
print(frame_list[0])
nfr = 40
t_list = [i/(nfr-1) for i in range(nfr)]
print(t_list)
# Fonction pour mettre à jour l'animation
def update(frame):
    plt.tight_layout()
    
    if -71 <= frame <= -10:
        t = t_list[0]
        interpolated_view = interpolate_view(view_start, view_end, t)
        change_view(interpolated_view)
        print(t_list)
        t_list.pop(0)

    if -10 < frame <= 0:
        ax.axis('on')
        opacity = calculate_opacity(frame)
        print("op",opacity)
        #set_axis_properties(ax, 1-opacity)
        ax.plane_surface.set_visible(False)
        ax.plane_surface = ax.plot_surface(x0, y0, z0, color='gray', alpha=opacity)
        if ax.hyperplan is not None:
            ax.hyperplan.set_visible(False)

    if frame > 0:
        ax.plane_surface.set_visible(False)
        ax.axis('on')
        interpolated_positions = calculate_interpolated_positions(X, X_3d, 0.45*frame)  # Nous utilisons X_3d pour les positions finales
        interpolated_positions2 = calculate_interpolated_positions(X, X_3d, frame)
        # Mettre à jour les positions des points de données
        scatter._offsets3d = (interpolated_positions[:, 0], interpolated_positions[:, 1], interpolated_positions[:, 2])
    
    
    # Ajouter un plan séparateur dans les dernières frames de l'animation
    if frame >= 85:
        #t_list = [i/(21) for i in range(21)]
        # Séparer les données par classe
        class_1_indices = np.where(y == 0)[0]
        class_2_indices = np.where(y == 1)[0]
        # Choisir un point dans chaque classe
        point_1 = interpolated_positions2[class_1_indices[0]]
        point_2 = interpolated_positions2[class_2_indices[0]]
        # Calculer le vecteur normal au plan séparateur
        normal_vector = point_2 - point_1
        # Normaliser le vecteur
        normal_vector /= np.linalg.norm(normal_vector)
        # Définir le plan séparateur à partir du vecteur normal
        xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, 2), np.linspace(-1.5, 1.5, 2))
        zz = (-normal_vector[0] * xx - normal_vector[1] * yy) / normal_vector[2]
        # Tracer le plan séparateur
        if 85 <= frame <= 89:
            ax.hyperplan = ax.plot_surface(xx, yy, zz, color='gray', alpha=0.10, label="Hyperplane")#ax.plot_surface(xx, yy, zz, color='#3b4a71',alpha = 0.5, label="Hyperplane")
    
    return scatter,
    

# Créer l'animation
plt.tight_layout()
ani = FuncAnimation(fig, update, frames=frame_list, interval=30)


ani.save('ktrick2.gif', writer='pillow', dpi=300)

"""
for i, frame in enumerate(np.linspace(-10, 100, 100)):
    update(frame)
    if i<=9:
        plt.savefig(f"gif_saves2/frame_0{i}.png",dpi=300, format='png', bbox_inches='tight')
    else:
        plt.savefig(f"gif_saves2/frame_{i}.png",dpi=300, format='png', bbox_inches='tight')

"""
plt.show()