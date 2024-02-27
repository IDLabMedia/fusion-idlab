import numpy as np
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf
from comprint.code.splicebuster.noiseprint.noiseprint_blind_concat import genMappFloat
from comprint.code.splicebuster.noiseprint.post_em import EMgu_img, getSpamFromNoiseprint
from comprint.code.splicebuster.noiseprint.noiseprint import genNoiseprint

# Comprint imports:
import comprint.code.network as comprint_network

# Generic imports:
from forgery_detection_generic import save_heatmap_to_file, save_image_to_npz_file, save_fingerprint_to_file, save_noiseprint_to_file, read_image_from_npz_file, load_image

import tensorflow.python.util.deprecation as deprecation
import os
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

MAX_INT32 = np.int64(2147483647)
def check_image_size(img):
    img_size = np.int64(img.shape[0]) * np.int64(img.shape[1]) * np.int64(64) # Found 64 experimentally (due to error message with underflowed value)
    if img_size > MAX_INT32:
        print('Image size is too big: %s' % str(img.shape))
        return False
    return True

def calculate_spam_em_heatmap(fingerprints, img_gray):
    spams = []
    for fingerprint in fingerprints:
        spam, valid, range0, range1, imgsize = getSpamFromNoiseprint(fingerprint, img_gray)
        spams.append(spam)

        if np.sum(valid) < 50:
            print("Sum valid less than 50: %d" % np.sum(valid))

    concatenated_spam = np.concatenate(tuple(spams), axis=2)

    heatmap, other = EMgu_img(concatenated_spam, valid, extFeat = range(32), seed = 0, maxIter = 100, replicates = 10, outliersNlogl = 42)
    
    range0  = range0.flatten()
    range1  = range1.flatten()
    #imgsize = imgsize.flatten()
    heatmap = genMappFloat(heatmap, valid, range0, range1, imgsize)
    return heatmap

def noiseprint(input_file_path, output_file_fingerprint_path, output_file_heatmap_path, model_path):
    # Load image
    img = load_image(input_file_path, channels=1).astype(np.float32)

    img_gray = img / 256.0 # For heatmap, later on

    try:
        QF = jpeg_qtableinv(input_file_path)
        print('QF=', QF)
    except:
        QF = 200

    # Extract fingerprint
    t = time()
    fingerprint = genNoiseprint(img_gray, QF, model_path)
    print("Fingerprint extracted in %.2f sec." % (time() - t))

    if output_file_fingerprint_path:
        save_noiseprint_to_file(fingerprint, output_file_fingerprint_path)
        #save_image_to_file(fingerprint, output_file_fingerprint_path)

        # Also save as npz for potential subsequent comprint+noiseprint fusion
        save_image_to_npz_file(fingerprint, output_file_fingerprint_path)

    # Heatmap with EM
    t = time()
    heatmap = calculate_spam_em_heatmap([fingerprint], img_gray)
    print("Heatmap extracted in %.2f sec." % (time() - t))
    save_heatmap_to_file(heatmap, output_file_heatmap_path)

    save_image_to_npz_file(heatmap, output_file_heatmap_path)

def comprint(input_file_path, output_file_fingerprint_path, output_file_heatmap_path, model, use_gpu=None):
    if use_gpu is None or not use_gpu:
        slide = 512
        device = '/CPU:0'
    else:
        slide = 1024
        device = '/GPU:0'
    with tf.device(device):
        # Load model if not loaded before
        if isinstance(model, str):
            model_network = comprint_network.Siamese_Network()
            model_network.build([None, None, None, 1])
            model_network.load_weights(model).expect_partial() 
            model = model_network

        # Load image
        img = load_image(input_file_path, channels=1).astype(np.float32)
        img_gray = img / 256.0 # For heatmap, later on

        # Rescale
        img = (img - 127.5) * 1./255 # Rescaling done during training of comprint method

        # Extract fingerprint
        t = time()
        fingerprint = comprint_tiled(img, model, slide=slide)
        print("Fingerprint extracted in %.2f sec." % (time() - t))

    if output_file_fingerprint_path:
        save_fingerprint_to_file(fingerprint, output_file_fingerprint_path)
        #save_image_to_file(fingerprint, output_file_fingerprint_path)

    # Rescale
    fingerprint = (fingerprint-np.mean(fingerprint))*(1/np.var(fingerprint))

    # Also save as npz for potential subsequent comprint+noiseprint fusion
    if output_file_fingerprint_path:
        save_image_to_npz_file(fingerprint, output_file_fingerprint_path)

    # Heatmap with EM
    t = time()
    heatmap = calculate_spam_em_heatmap([fingerprint], img_gray)
    print("Heatmap extracted in %.2f sec." % (time() - t))
    save_heatmap_to_file(heatmap, output_file_heatmap_path)

    save_image_to_npz_file(heatmap, output_file_heatmap_path)


def comprint_tiled(img, model, slide=1024):
    overlap = 34
    largeLimit = slide*slide + 1
    if img.shape[0] * img.shape[1] > largeLimit:
        res = np.zeros((img.shape[0],img.shape[1]), np.float32)
        for index0 in range(0, img.shape[0], slide):
            index0start = index0 - overlap
            index0end   = index0 + slide + overlap
            
            for index1 in range(0, img.shape[1], slide):
                index1start = index1 - overlap
                index1end   = index1 + slide + overlap
                clip = img[max(index0start, 0): min(index0end,  img.shape[0]), max(index1start, 0): min(index1end,  img.shape[1])]
                resB = model.predict(clip[None, ..., None], verbose=0)
                resB = np.squeeze(resB)

                if index0 > 0:
                    resB = resB[overlap:, :]
                if index1 > 0:
                    resB = resB[:, overlap:]
                resB = resB[:min(slide, resB.shape[0]), :min(slide, resB.shape[1])]
                
                res[index0: min(index0+slide, res.shape[0]), index1: min(index1+slide, res.shape[1])] = resB
    else:
        # Add batch dimension and channels dimension
        img = img[None, ...,None]
        res = model.predict(img, verbose=0)
        res = np.squeeze(res)
    return res


# Assumes the comprint and noiseprint fingerprints were previously extracted and saved as npz files
def comprint_plus_noiseprint(input_file_path, output_file_comprint_fingerprint_path, output_file_noiseprint_fingerprint_path, output_file_heatmap_path):
    # Load image
    img = load_image(input_file_path, channels=1).astype(np.float32)

    img_gray = img / 256.0 # For heatmap, later on

    # Read fingerprints from file
    comprint_fingerprint = read_image_from_npz_file(output_file_comprint_fingerprint_path)
    noiseprint_fingerprint = read_image_from_npz_file(output_file_noiseprint_fingerprint_path)

    # Heatmap with EM
    t = time()
    heatmap = calculate_spam_em_heatmap([comprint_fingerprint, noiseprint_fingerprint], img_gray)
    print("Heatmap extracted in %.2f sec." % (time() - t))
    save_heatmap_to_file(heatmap, output_file_heatmap_path)

    save_image_to_npz_file(heatmap, output_file_heatmap_path)