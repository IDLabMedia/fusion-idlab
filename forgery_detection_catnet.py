import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)
catnet_path = './CAT-Net'
if catnet_path not in sys.path:
    sys.path.insert(1, catnet_path)

import argparse
import torch
#print("Torch device: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank
from Splicing.data.data_core import SplicingDataset as splicing_dataset
import cv2 as cv
import seaborn as sns; sns.set_theme()

from forgery_detection_generic import save_heatmap_to_file, save_image_to_npz_file
import cv2


# model = path to model
def catnet(input_file_path, output_file_heatmap_path, model, use_gpu=True):        
    args = argparse.Namespace(cfg='./CAT-Net/experiments/CAT_full.yaml', opts=[
        'TEST.MODEL_FILE', model,
        'TEST.FLIP_TEST', 'False',
        'TEST.NUM_SAMPLES', '0'
        ])
    update_config(config, args)

    file = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1,
        mode='Single file', read_from_jpeg=True, filename=input_file_path)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    data = torch.utils.data.DataLoader(
        file,
        batch_size=1,  # must be 1 to handle arbitrary input sizes
        shuffle=False,  # must be False to get accurate filename
        num_workers=0,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=file.class_weights)

    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=file.class_weights)
    if use_gpu:
        criterion = criterion.cuda()

    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
        print('=> loading model from {}'.format(model_state_file))
    else:
        raise ValueError("Model file is not specified.")

    model = FullModel(model, criterion)
    if use_gpu:
        checkpoint = torch.load(model_state_file)
    else:
        checkpoint = torch.load(model_state_file, map_location=torch.device("cpu"))

    model.model.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        gpus = list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus)
        model = model.cuda()

    with torch.no_grad():
        for image, label, qtable in data:
            size = label.size()
            label = label.long()
            if use_gpu:
                image = image.cuda()
                label = label.cuda()
            model.eval()
            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()
            heatmap = pred
            
            save_heatmap_to_file(heatmap, output_file_heatmap_path, vmin=0, vmax=1)

            # Resize heatmap to original image size (e.g., 1024x682 became 1240x833)
            heatmap = cv2.resize(heatmap, dsize=cv2.imread(input_file_path).shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
            save_image_to_npz_file(heatmap, output_file_heatmap_path)
