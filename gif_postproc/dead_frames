from PIL import Image, ImageSequence

def add_dead_frames(input_gif, output_gif, num_dead_frames=10):
    # Ouvrir le GIF existant
    gif = Image.open(input_gif)
    
    # Extraire les frames
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    
    # Première et dernière frame
    first_frame = frames[0]
    last_frame = frames[-1]
    
    # Ajouter des frames mortes au début et à la fin
    dead_start = [first_frame] * num_dead_frames
    dead_end = [last_frame] * num_dead_frames
    
    # Créer la nouvelle liste de frames
    new_frames = dead_start + frames + dead_end
    
    # Enregistrer le nouveau GIF avec des frames mortes
    new_frames[0].save(
        output_gif, save_all=True, append_images=new_frames[1:], loop=0, duration=gif.info['duration']
    )

# Exemple d'utilisation
input_gif = 'ktrick2-ezgif.gif'
output_gif = 'ktrick2-ezgif_dead_frames.gif'
add_dead_frames(input_gif, output_gif, num_dead_frames=50)
