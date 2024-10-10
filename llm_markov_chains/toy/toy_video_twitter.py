import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.gridspec as gridspec
from utils import all_possible, all_possible_small  # Ensure these functions are correctly defined and imported

# ------------------------ Configuration Variables ------------------------

# Subplot sizes (in inches)
FIG_WIDTH = 16
FIG_HEIGHT = 9

# Animation settings
INTERVAL = 500  # Time between frames in milliseconds
REPEAT_DELAY = 2000  # Delay before repeating the animation

# Colormap configuration using seaborn's cubehelix_palette
BASE_CMAP = sns.cubehelix_palette(start=0, rot=0.5, dark=0.3, light=0.8, as_cmap=True)

# ------------------------ Helper Functions ------------------------

def save_subplot_as_image(fig, ax, filename, dpi=300):
    """
    Save a single subplot (ax) from a figure (fig) as a separate image file.

    Parameters:
    - fig: matplotlib figure object
    - ax: matplotlib axes object to save
    - filename: string, path to save the image (e.g., 'Matrix_A.png' or 'Matrix_A.pdf')
    - dpi: resolution of the saved image
    """
    # Hide all other axes
    for other_ax in fig.axes:
        if other_ax != ax:
            other_ax.set_visible(False)

    # Adjust layout to fit the single visible subplot
    fig.tight_layout()

    # Save the figure with only the desired Axes visible
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')

    # Show all axes again
    for other_ax in fig.axes:
        other_ax.set_visible(True)


def random_one_hot(p):
    """
    Generates a one-hot vector of size 1 x p with a single 1 at a random position.

    Parameters:
    - p (int): The size of the vector.

    Returns:
    - np.ndarray: A 1 x p one-hot vector.
    """
    b = np.zeros((1, p))
    idx = np.random.randint(0, p)
    b[0, -4] = 1.0
    return b

def mask_zeros(matrix):
    """
    Replace zeros in the matrix with NaN for masking in imshow.

    Parameters:
    - matrix (np.ndarray): The input matrix.

    Returns:
    - np.ndarray: The matrix with zeros replaced by NaN.
    """
    return np.where(matrix == 0, np.nan, matrix)

def mask_nans(matrix):
    """
    Mask only the NaN values in the matrix for imshow.

    Parameters:
    - matrix (np.ndarray): The input matrix.

    Returns:
    - np.ma.MaskedArray: The masked matrix with NaNs masked.
    """
    return np.ma.masked_invalid(matrix)

# ------------------------ Load Data ------------------------

# Load the matrix A from 'data.pkl'
with open('data.pkl', 'rb') as file:
    A = pickle.load(file)  # Assuming A is a 2D numpy array of shape (p, p)

# Validate that A is a square matrix
if not isinstance(A, np.ndarray):
    raise TypeError("Matrix A must be a NumPy array.")
if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError("Matrix A must be a square matrix of shape (p, p).")

p = A.shape[1]  # Dimension size

# Generate vector b as a one-hot vector
b = random_one_hot(p)

# Compute vector c as the product of b and A
c = np.dot(b, A)  # Shape: (1, p)

# ------------------------ Prepare Data for Visualization ------------------------

# Mask zeros in A and b
A_masked = mask_zeros(A)
b_masked = mask_zeros(b)

# Initialize c_display with all NaNs (blank)
c_display = np.full(c.shape, np.nan)

# Determine vmin and vmax across A, b, and c for consistent color mapping
# Exclude NaNs (which represent zeros) in the calculation
combined_data = np.concatenate((A.flatten(), b.flatten(), c.flatten()))
vmin = np.nanmin(combined_data)
vmax = np.nanmax(combined_data)

# Create a custom colormap that treats NaN as white
cmap_with_blank = BASE_CMAP.copy()
cmap_with_blank.set_bad(color='white')


# Create a normalization object based on vmin and vmax
norm = Normalize(vmin=vmin, vmax=vmax)

# ------------------------ Set Up the Figure and Subplots ------------------------

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[14, 1], wspace=0.3, hspace=0.3)

# Top-Left: Colorbar
ax_colorbar = fig.add_subplot(gs[0, 0])
ax_colorbar.axis('off')  # Hide the axis

# Generate sequences for labels
vocab_size = 2
context_length = 3

all_seqs = list(all_possible_small(vocab_size, context_length)) + list(all_possible(vocab_size, context_length))
print(f"Total sequences: {len(all_seqs)}")  # For debugging purposes

# Ensure that the number of sequences matches the matrix dimensions
if len(all_seqs) != p:
    raise ValueError(f"Number of sequences ({len(all_seqs)}) does not match the dimension of A ({p}).")

# Top-Right: Matrix A
ax_A = fig.add_subplot(gs[0, 1])
im_A = ax_A.imshow(A_masked, cmap=cmap_with_blank, norm=norm, aspect='equal', origin='upper')

# Set tick positions and labels
positions = np.arange(len(all_seqs))

# Align ticks to the center of each cell by adding 0.5, and put them to the top
ax_A.xaxis.set_ticks_position('top')
ax_A.set_xticks(positions +0.01)
ax_A.set_yticks(positions +0.01)

# Set tick labels
ax_A.set_xticklabels(all_seqs, rotation=90, fontsize=12)
ax_A.set_yticklabels(all_seqs, fontsize=12)  # Reverse y-axis labels to match 'upper' origin

# Bottom-Left: Vector b
ax_b = fig.add_subplot(gs[1, 0])
b_image = mask_zeros(b)  # Shape: (1, p)
im_b = ax_b.imshow(b_image, cmap=cmap_with_blank, norm=norm, aspect='auto')

# Set tick positions and labels for b
ax_b.set_xticks(positions +0.01)
ax_b.set_yticks([])  # No y-axis ticks

ax_b.set_xticklabels(all_seqs, rotation=90, fontsize=12, fontweight='bold')

# Bottom-Right: Vector c
ax_c = fig.add_subplot(gs[1, 1])
im_c = ax_c.imshow(c_display, cmap=cmap_with_blank, norm=norm, aspect='auto')

# Set tick positions and labels for c
ax_c.set_xticks(positions +0.01)
ax_c.set_yticks([])  # No y-axis ticks

ax_c.set_xticklabels(all_seqs, rotation=90, fontsize=12)

# ------------------------ Add Unified Colorbar ------------------------

# Create a ScalarMappable for the colorbar
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cmap_with_blank, norm=norm)
sm.set_array([])  # Dummy array for the ScalarMappable

# Add the colorbar to the top-left subplot
"""
cbar = fig.colorbar(sm, cax=ax_colorbar, orientation='vertical')
cbar.set_label('Value', fontsize=14)
"""

# ------------------------ Animation Function ------------------------

def update(frame):
    """
    Update function for the animation.
    At each frame, reveal the frame-th coefficient of vector c.

    Parameters:
    - frame (int): The current frame number.

    Returns:
    - list: A list containing the updated image object.
    """
    # Create a copy of c to avoid modifying the original
    c_temp = c.copy()

    if frame < p:
        # Mask coefficients beyond the current frame
        mask = np.zeros(c.shape, dtype=bool)
        mask[0, frame + 1:] = True  # Mask coefficients after the current frame
        c_temp = np.where(mask, np.nan, c_temp)
    else:
        # If frame exceeds p, show all coefficients
        pass  # c_temp remains as c

    # Mask zeros to keep them as blank
    c_masked = mask_zeros(c_temp)

    # Update the image data for vector c
    im_c.set_data(c_masked)
    
    frame_filename = f'frame_{frame+1}.png'
    if frame == p-1:
        #save_subplot_as_image(fig, ax_A, f'C:/Users/dofel/Documents/FIGURE DE FOU/Matrix_A5_{frame_filename}')
        save_subplot_as_image(fig, ax_b, f'C:/Users/dofel/Documents/FIGURE DE FOU/Vector_b5_gras_{frame_filename}')
        #save_subplot_as_image(fig, ax_c, f'C:/Users/dofel/Documents/FIGURE DE FOU/Vector_c5_{frame_filename}')

    return [im_c]

# ------------------------ Create the Animation ------------------------

ani = animation.FuncAnimation(
    fig,
    update,
    frames=p,  # Number of frames equals the number of coefficients
    interval=INTERVAL,
    repeat=True,
    blit=True,
    repeat_delay=REPEAT_DELAY
)

# ------------------------ Display the Animation ------------------------

plt.show()

# ------------------------ (Optional) Save the Animation ------------------------

# To save the animation as a GIF or MP4, uncomment the following lines
# Ensure that you have 'imagemagick' or 'ffmpeg' installed respectively

# ani.save('matrix_product_animation.gif', writer='imagemagick', fps=1)
#ani.save('C:/Users/dofel/Documents/FIGURE DE FOU/matrix_product_animation_b4.mp4', writer='ffmpeg', fps=1)
