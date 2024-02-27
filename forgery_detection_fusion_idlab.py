import fusion_idlab
from time import time

# Generic imports:
from forgery_detection_generic import save_heatmap_to_file

# Assumes the heatmaps were previously extracted and saved as npz files
def run_fusion_idlab(input_file_path,
           cat_net_image_path, comprint_noiseprint_image_path, adq1_image_path, blk_image_path, dct_image_path, cagi_image_path, comprint_image_path, noiseprint_image_path,
           output_file_heatmap_path, model, use_gpu=None):
    # Heatmap with EM
    t = time()
    heatmap = fusion_idlab.get_prediction(model,
                                          cat_net_image_path, comprint_noiseprint_image_path, adq1_image_path, blk_image_path, dct_image_path, cagi_image_path, comprint_image_path, noiseprint_image_path,
                                          input_file_path)
    print("Heatmap extracted in %.2f sec." % (time() - t))
    save_heatmap_to_file(heatmap, output_file_heatmap_path)