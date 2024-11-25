import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Séquence et nombre d'états
sequence = [1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 1, 0]
num_states = 3

# Initialisation de la matrice de transition
transition_matrix = np.zeros((num_states, num_states))

# Historique des matrices pour l'animation
matrix_history = []

# Construire les matrices étape par étape
for i in range(len(sequence) - 1):
    current_state = sequence[i]
    next_state = sequence[i + 1]
    transition_matrix[current_state, next_state] += 1
    # Normalisation pour obtenir les probabilités
    normalized_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    matrix_history.append(normalized_matrix.copy())

# Animation avec les valeurs décimales affichées au lieu de couleurs

fig, ax = plt.subplots()

# Créer une matrice vide pour afficher les valeurs
matrix_table = ax.table(cellText=np.round(matrix_history[0], 2),
                        cellLoc='center',
                        loc='center',
                        colLabels=[f"État {i}" for i in range(num_states)],
                        rowLabels=[f"État {i}" for i in range(num_states)])
ax.axis('tight')
ax.axis('off')

def update_table(frame):
    for (i, j), val in np.ndenumerate(np.round(matrix_history[frame], 2)):
        matrix_table[i, j]._text.set_text(f"{val:.2f}")
    ax.set_title(f"Transition {frame + 1}: {sequence[frame]} → {sequence[frame + 1]}")

ani = FuncAnimation(fig, update_table, frames=len(matrix_history), interval=500)

# Sauvegarder l'animation pour consultation ou affichage direct
ani.save('freq_vs_llmicl/markov_chain_animation.gif', writer='imagemagick')


# Extraction des valeurs des matrices pour chaque étape
matrix_values_history = [matrix.tolist() for matrix in matrix_history]

import pandas as pd

# Convertir la liste des matrices en un DataFrame pour une meilleure visualisation
steps = [f"Step {i+1}" for i in range(len(matrix_values_history))]
df_matrices = pd.DataFrame({
    "Step": steps,
    "Matrix": matrix_values_history
})

import pickle

# Sauvegarder les valeurs des matrices dans un fichier .pkl
pkl_file_path = 'freq_vs_llmicl/markov_chain_matrices.pkl'
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(matrix_values_history, pkl_file)

pkl_file_path

