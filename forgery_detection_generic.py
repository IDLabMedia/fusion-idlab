import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_heatmap_to_file(heatmap, output_image_path, vmin=None, vmax=None):
    # Save heatmap to file
    ensure_dir(output_image_path)
    fig = plt.figure()
    if vmin is None:
        vmin = np.nanmin(heatmap)
    if vmax is None:
        vmax = np.nanmax(heatmap)
    plt.imsave(output_image_path, heatmap, vmin=vmin, vmax=vmax, cmap='jet', format='png')
    plt.close(fig)

def save_fingerprint_to_file(fingerprint, output_file_fingerprint_path):
    ensure_dir(output_file_fingerprint_path)
    fig = plt.figure()
    plt.imsave(output_file_fingerprint_path, fingerprint, cmap='tab20b', format='png')
    plt.close(fig)

def save_image_to_npz_file(image, output_file_path):
    np.savez_compressed("%s.npz" % (output_file_path), image=image)

def read_image_from_npz_file(output_file_path):
    image = np.load("%s.npz" % (output_file_path))['image']
    return image 

def save_noiseprint_to_file(fingerprint, output_file_fingerprint_path):
    ensure_dir(output_file_fingerprint_path)
    fig = plt.figure()
    vmin = np.min(fingerprint[34:-34,34:-34])
    vmax = np.max(fingerprint[34:-34,34:-34])
    plt.imsave(output_file_fingerprint_path, fingerprint.clip(vmin,vmax), vmin=vmin, vmax=vmax, cmap='tab20b', format='png')
    plt.close(fig)

def save_image_to_file(image, output_file_fingerprint_path):
    ensure_dir(output_file_fingerprint_path)
    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_file_fingerprint_path, dpi=250, pad_inches=0,  bbox_inches='tight')
    plt.close(fig)

def load_image(input_file_path, channels=1):
    mode = "L" if channels == 1 else "RGB" # channels == 3
    image = Image.open(input_file_path)
    image = image.convert(mode=mode)
    img = np.array(image)   
    image.close()
    return img