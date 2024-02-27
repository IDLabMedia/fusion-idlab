import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import pyIFD.DCT
import pyIFD.ADQ1
import pyIFD.BLK
import pyIFD.CAGI

from forgery_detection_generic import ensure_dir, save_heatmap_to_file, save_fingerprint_to_file, save_image_to_npz_file, read_image_from_npz_file, save_noiseprint_to_file, save_image_to_file, load_image

def dct(input_image_path, output_image_path):
    heatmap = pyIFD.DCT.DCT(input_image_path)
    save_heatmap_to_file(heatmap, output_image_path)

    heatmap = cv2.resize(heatmap, dsize=cv2.imread(output_image_path).shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    save_image_to_npz_file(heatmap, output_image_path)

def adq1(input_image_path, output_image_path):
    file_name, file_extension = os.path.splitext(input_image_path)
    heatmap = pyIFD.ADQ1.detectDQ(input_image_path)
    if file_extension == ".jpg" or file_extension == ".jpeg":
        heatmap = heatmap[0]
    save_heatmap_to_file(heatmap, output_image_path)

    heatmap = cv2.resize(heatmap, dsize=cv2.imread(output_image_path).shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    save_image_to_npz_file(heatmap, output_image_path)

def blk(input_image_path, output_image_path):
    heatmap = pyIFD.BLK.GetBlockGrid(input_image_path)[0]
    save_heatmap_to_file(heatmap, output_image_path)

    heatmap = cv2.resize(heatmap, dsize=cv2.imread(output_image_path).shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    save_image_to_npz_file(heatmap, output_image_path)

def cagi(input_image_path, output_image_path, output_image_path_inv=""):
    heatmap, heatmap_inv = pyIFD.CAGI.CAGI(input_image_path)
    save_heatmap_to_file(heatmap, output_image_path)
    if output_image_path_inv:
        save_heatmap_to_file(heatmap_inv, output_image_path_inv)

    heatmap = cv2.resize(heatmap, dsize=cv2.imread(output_image_path).shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    save_image_to_npz_file(heatmap, output_image_path)