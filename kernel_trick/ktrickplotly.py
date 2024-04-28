import numpy as np
import plotly.graph_objs as go
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from plotly.subplots import make_subplots

# Générer des données circulaires non linéairement séparables en 2D
np.random.seed(0)
X, y = make_circles(n_samples=50, factor=0.5, noise=0.1)

# Ajouter une colonne de zéros à la matrice X pour obtenir une représentation en 3D
X = np.hstack([X, np.zeros((X.shape[0], 1))])

# Initialiser le modèle SVM avec un noyau gaussien
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1))
svm.fit(X[:, :2], y)  # Nous utilisons uniquement les deux premières colonnes de X pour l'entraînement

# Transformer les données en 3D avec le kernel trick
X_3d = svm.decision_function(X[:, :2]).reshape(-1, 1)  # Mettre à jour la colonne z avec les valeurs du kernel trick

# Créer la figure
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Tracer les points de données 2D initiaux
scatter_trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=np.zeros_like(X[:, 0]), mode='markers',
                           marker=dict(color=y, colorscale='jet', size=8))
fig.add_trace(scatter_trace, row=1, col=1)

# Calculer les positions interpolées des points de données
def calculate_interpolated_positions(X, X_3d, frame):
    interpolated_positions = X + (X_3d - X) * (frame / 100)
    return interpolated_positions




# Créer l'animation
frames = []
for frame in np.linspace(0, 100, 100):
    interpolated_positions = calculate_interpolated_positions(X, X_3d, frame)  # Nous utilisons X_3d pour les positions finales
    scatter_trace = go.Scatter3d(x=interpolated_positions[:, 0], y=interpolated_positions[:, 1],
                                 z=interpolated_positions[:, 2], mode='markers',
                                 marker=dict(color=y, colorscale='jet', size=8))
    frames.append(go.Frame(data=[scatter_trace],name=str(frame)))
        

# Ajouter les frames à l'animation
fig.frames = frames


# Définir les limites des axes pour une meilleure visualisation
fig.update_layout(scene=dict(xaxis=dict(range=[-2, 2], gridcolor='lightgray', showbackground=True, showticklabels=False, showgrid=False),
                             yaxis=dict(range=[-2, 2], gridcolor='lightgray', showbackground=True, showticklabels=False, showgrid=False),
                             zaxis=dict(range=[-2, 2], gridcolor='lightgray', showbackground=True, showticklabels=False, showgrid=False),
                             aspectratio=dict(x=1, y=1, z=1)),
                             plot_bgcolor='rgba(255,255,255,0)',
                  scene_camera=dict(eye=dict(x=-1.25, y=-1.25, z=0.25)),
                  
                  scene_aspectmode='cube')

fig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Trigger the Kernel Trick !",
                                                                method="animate",
                                                                args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                                 fromcurrent=True)])])])

fig.show()
fig.write_html('kernel_trick.html')