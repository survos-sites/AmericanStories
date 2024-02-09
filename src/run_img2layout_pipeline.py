from stages.images_to_layouts import get_onnx_input_name, letterbox, non_max_suppression
# from stages.pdfs_to_images import pdfs_to_images
from effocr.engines.yolov8_ops import non_max_suppression as non_max_supression_yolov8
from generate_manifest import *
from utils.ca_metadata_utils import *

import os
import requests
import json
import faiss
import argparse
from tqdm import tqdm
from timeit import default_timer as timer
import torch
import onnx
import onnxruntime as ort
import numpy as np
import logging
from PIL import Image, ImageDraw
import cv2
import gc
import psutil
import pkg_resources
from torchvision.ops import nms
from torchvision import transforms
import random
import multiprocessing
import time
from math import floor, ceil
from pytorch_metric_learning.utils.inference import FaissKNN
from pytorch_metric_learning.utils import common_functions as c_f

'''
Avoid rate limiting:

- wget images
- stagger jobs
- different subnets -- randomize ips on the subnets?
- Different vns for different regions, etc?
- Metadata readd?? - slow things down
- user agents -- for metadata too? or dump if wgetting
'''

LAYOUT_TYPES_TO_EFFOCR = ['article', 'author', 'headline', 'image_caption']
IMG_FILE_EXS = ('jpg', 'png', 'jp2')
LEGIBLE_LABEL_MAP = ['Legible', 'Questionable', 'Illegible']
LAYOUT_COLOR_MAP = {'article': 'blue', 'headline': 'red', 'cartoon_or_advertisement': 'orange'}

USER_HEADERS = ['Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.94 Chrome/37.0.2062.94 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
                                    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/600.8.9 (KHTML, like Gecko) Version/8.0.8 Safari/600.8.9',
                                    'Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4',
                                    'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10240',
                                    'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
                                    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                                    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_4) AppleWebKit/600.7.12 (KHTML, like Gecko) Version/8.0.7 Safari/600.7.12',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:40.0) Gecko/20100101 Firefox/40.0',
                                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.8.9 (KHTML, like Gecko) Version/7.1.8 Safari/537.85.17',
                                    'Mozilla/5.0 (iPad; CPU OS 8_4 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H143 Safari/600.1.4',
                                    'Mozilla/5.0 (iPad; CPU OS 8_3 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12F69 Safari/600.1.4']

def ord_str_to_word(w):
    return ''.join([chr(int(c)) for c in w.split('_')])

def get_crops_from_layout_image(image):
    im_width, im_height = image.size[0], image.size[1]
    if im_height <= im_width * 2:
        return [image]
    else:
        y0 = 0
        y1 = im_width * 2
        crops = []
        while y1 <= im_height:
            crops.append(image.crop((0, y0, im_width, y1)))
            y0 += int(im_width * 1.5)
            y1 += int(im_width * 1.5)
        
        crops.append(image.crop((0, y0, im_width, im_height)))
        return crops
    
def readjust_line_detections(line_preds, orig_img_width):
    y0 = 0
    dif = int(orig_img_width * 1.5)
    all_preds, final_preds = [], []
    for j in range(len(line_preds)):
        preds, probs, labels = line_preds[j]
        for i, pred in enumerate(preds):
            all_preds.append((pred[0], pred[1] + y0, pred[2], pred[3] + y0, probs[i]))
        y0 += dif
    
    all_preds = torch.tensor(all_preds)
    if all_preds.dim() > 1:
        keep_preds = nms(all_preds[:, :4], all_preds[:, -1], iou_threshold=0.15)
        filtered_preds = all_preds[keep_preds, :4]
        filtered_preds = filtered_preds[filtered_preds[:, 1].sort()[1]]
        for pred in filtered_preds:
            x0, y0, x1, y1 = torch.round(pred)
            x0, y0, x1, y1 = x0.item(), y0.item(), x1.item(), y1.item()
            final_preds.append((x0, y0, x1, y1))
        return final_preds
    else:
        return []

# Get all layout predictions for an image
def get_layout_predictions(layout_session, label_map_layout, ca_img, layout_output, f_idx, input_name, backend='yolo'):
    #finetuned yolov5 ONNX model
    # Resize and reshape image to fit model input
    im = letterbox(ca_img, (1280, 1280), auto=False)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.expand_dims(np.ascontiguousarray(im), axis = 0).astype(np.float32) / 255.0  # contiguous

    layout_predictions = layout_session.run(
        None,
        {input_name: im}
    )

    layout_predictions = torch.from_numpy(layout_predictions[0])
    if backend == 'yolo':
        layout_predictions = non_max_suppression(layout_predictions, conf_thres = 0.05, iou_thres=0.01, max_det=300, agnostic = True)[0] 
        print(layout_predictions.size())    
    elif backend == 'yolov8':
        layout_predictions = non_max_supression_yolov8(layout_predictions, conf_thres = 0.01, iou_thres=0.1, max_det=2000, agnostic = True)[0]
    
    layout_bboxes, layout_probs, layout_labels = layout_predictions[:, :4], layout_predictions[:, -2], layout_predictions[:, -1]

    crops_for_effocr = []
    layout_img = Image.fromarray(ca_img)
    im_width, im_height = layout_img.size[0], layout_img.size[1]
    
    # Precompute ratios and translations to rescale bounding boxes to original image size
    if im_width > im_height:
        w_ratio = 1280
        h_ratio = (im_width / im_height) * 1280
        w_trans = 0
        h_trans = 1280 * ((1 - (im_height / im_width)) / 2)
    else:
        h_trans = 0
        h_ratio = 1280
        w_trans = 1280 * ((1 - (im_width / im_height)) / 2)
        w_ratio = 1280 * (im_width / im_height)

    # Set up for drawing bounding boxes on image if requested
    if layout_output:
        draw = ImageDraw.Draw(layout_img)

    # Iterate through predicted bounding boxes, cropping out each from the original image
    layout_crops = []
    for i, (line_bbox, pred_class) in enumerate(zip(layout_bboxes, layout_labels)):
        x0, y0, x1, y1 = torch.round(line_bbox) # Grab the bounding box coordinates on resized image
        # Convert coordinages to original image size (messy)
        x0, y0, x1, y1 = int(floor((x0.item() - w_trans) * im_width / w_ratio)), int(floor((y0.item() - h_trans) * im_height / h_ratio)), \
                        int(ceil((x1.item() - w_trans) * im_width / w_ratio)), int(ceil((y1.item() - h_trans) * im_height  / h_ratio))

        # Crop out image and append to list of layout crops
        layout_crop = layout_img.crop((x0, y0, x1, y1))
        layout_crops.append((pred_class, (x0, y0, x1, y1), layout_crop))
        
        # Append chunked crop to a separate list of crops to EffOCR if in one of the desired types
        if label_map_layout[int(pred_class.item())] in LAYOUT_TYPES_TO_EFFOCR:
            crops = get_crops_from_layout_image(layout_crop) # Chunk the crop if it's a poor aspect ratio for line detection
            for crop in crops:
                crops_for_effocr.append((i, crop))
            
        if layout_output: # optionally save layout bounding boxes to file
            draw.rectangle((x0, y0, x1, y1), outline=LAYOUT_COLOR_MAP.get(label_map_layout[int(pred_class.item())], 'black'), width=5)
            # layout_crop.save(os.path.join(layout_output, 'bbox_{}_{}_{}.jpg'.format(f_idx, i, label_map_layout[int(pred_class.item())])))
    
    if layout_output: # Save the image with drawn bounding boxes if requested
        layout_img.save(os.path.join(layout_output, f'layout_boxes_{f_idx}.jpg'))

    return crops_for_effocr, layout_crops

def get_onnx_input_name(model):
    input_all = [node.name for node in model.graph.input]
    input_initializer =  [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    return net_feed_input[0]

def main(args):

    # Save arguments
    output_save_path = args.output_save_path
    img_save_path = os.path.join(output_save_path, "images")
    checkpoint_path_layout = args.checkpoint_path_layout
    checkpoint_path_line = args.checkpoint_path_line

    label_map_path_layout = args.label_map_path_layout

    first_n = args.first_n
    layout_line_only = args.layout_line_only

    layout_output = args.layout_output
    manifest_path = args.manifest_path
    bbox_output = args.bbox_output
    layout_model_backend = args.layout_model_backend
    
    punc_padding = args.punc_padding

    # Create output directories
    os.makedirs(output_save_path, exist_ok=True)
    errors_log = []

    # Read in manifest of scans to process
    if os.path.isdir(manifest_path):
        filenames = [os.path.join(manifest_path, p) for p in os.listdir(manifest_path)]
    elif os.path.isfile(manifest_path):
        with open(args.manifest_path) as infile:
            filenames = infile.readlines()
    else:
        raise FileNotFoundError('Could not find manifest in {}'.format(manifest_path))
    
    # Create extra output directories for visualization
    if layout_output: os.makedirs(layout_output, exist_ok=True)

    # Truncate if requested
    if first_n:
        filenames = filenames[:first_n]

    # Import pdf converter if needed
    if any([f.endswith('.pdf') for f in filenames]):
        '''
        NOTE: PDF functionality is not included in the requirements.txt file because it tends to 
        create compatiblity problems on Azure machines. If you want to use the functionality, 
        first run
            `pip install pikepdf`
        in your environment. Then provide one or more pdf files (can be downloaded from 
        chronicling america or stored locally) in either your manifest or your directory. 
        '''
        from stages.pdfs_to_images import pdfs_to_images

    # Set up logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(output_save_path, 'ca_transcription.txt'), level=logging.INFO)
    logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.WARNING)
    logging.info('Transcribing {} files'.format(len(filenames)))
    
    # Load Layout Model
    layout_base_model = onnx.load(checkpoint_path_layout)
    layout_input_name = get_onnx_input_name(layout_base_model)
    del layout_base_model
    layout_inf_sess = ort.InferenceSession(checkpoint_path_layout)

    # Load layout label map
    with open(label_map_path_layout) as jf:
        label_map_data = json.load(jf)
        label_map_layout = {int(k): v for k, v in label_map_data.items()}
        del label_map_data
    
    img_download_session = requests.Session()

    # TODO: this should really be in an OOP format
    scan_time = 100
    pg_num = 0
    old_ed = ''
    for f_idx, f in enumerate(filenames):
        if scan_time < 5:
            time.sleep(10)
        scan_start_time = timer()
        gc.collect()
        #========== Fetch Metadata =============================================
        # start_time = timer()
        # try:
        #     if not os.path.isfile(f): # Only fetch if getting files from the Chronicling America
        #         lccn = f.strip().split('/')[-4]
        #         year_ed = f.strip().split('/')[-2]
        #         lccn_metadata = get_lccn_metadata(lccn)
        #         edition_metadata = get_edition_metadata(lccn, year_ed)
        #         page_num = find_page_number_from_filename(f.strip())
        #         if page_num is not None:
        #             scan_metadata = get_scan_metadata(lccn, year_ed[:8], year_ed[8:], page_num)
        #         else:
        #             scan_metadata = {}
        #         logging.info('Got scan metadata')

        #         metadata = {'lccn': lccn_metadata,
        #                     'edition': edition_metadata,
        #                     'page_number': page_num,
        #                     'scan': scan_metadata }
                    
        #         metadata_time = timer() - start_time
        #         logging.info('Fetch Metadata: {}'.format(metadata_time))
        #     else:
        #         metadata = {'page_number': 'na', 'scan_url': f, 'scan_ocr': 'na'}
        # except Exception as e:
        #     logging.error('Error fetching metadata: {}'.format(e))
        #     errors_log.append((f.strip(), 'Metadata', str(e)))
        #     scan_time = timer() - scan_start_time
        #     continue

        #========== CA Image Download =============================================
        metadata = {'page_number': 'na', 'scan_url': f, 'scan_ocr': 'na', 'scan':{}}
        logging.info(f.strip())

        start_time = timer()
        try:
            if os.path.isfile(f):
                if f.endswith(IMG_FILE_EXS):
                    ca_img = cv2.imread(f, cv2.IMREAD_COLOR)
                elif f.endswith('.pdf'):
                    pdfs_to_images(
                        source_path=f,
                        save_path=manifest_path,
                        data_source='na',
                        nested=False,
                        resize=False,
                        deskew=False
                    )
                    ca_img = cv2.imread(f[:-4] + '.jpg', cv2.IMREAD_COLOR)
                else:
                    print('Unknown file type for {}, only jpg, png, jp2, pdf supported!'.format(f))
                    continue
            else:
                # os.system(f'wget -O ca_img.jp2 {f.strip()}')
                # data = cv2.imread('ca_img.jp2')
                ca_batch_url = f.strip()
                lccn, reel, ed, scan = f.split('/')[-4:]

                if ed == old_ed: pg_num += 1
                else: pg_num = 1

                if len(str(pg_num)) == 1: pg_num_str = '0' + str(pg_num)
                else: pg_num_str = str(pg_num)
                date = ed[:8]
                year, month, day = date[:4], date[4:6], date[6:]
                ed_num = ed[8:]
                ca_url = f'https://chroniclingamerica.loc.gov/lccn/{lccn}/{year}-{month}-{day}/ed-{ed_num}/seq-{pg_num_str}.jp2'
                logging.info('Downloading image from {}'.format(ca_url))
                
                response = img_download_session.get(ca_url, headers = {'User-Agent': random.choice(USER_HEADERS)})
                data = response.content
                ca_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if ca_img is None:
                    logging.info('Image not found: {}'.format(f.strip()))
                    logging.info('Response code: {}'.format(response.status_code))
                    logging.info('Response: {}'.format(response))
                    logging.info('Response headers: {}'.format(response.headers))
                    logging.info('Response content: {}'.format(response.content))
                    logging.info('Response url: {}'.format(response.url))
                    logging.info('Response text: {}', response.text)
                    continue
                else:
                    metadata['scan']['height'] = ca_img.shape[0]
                    metadata['scan']['width'] = ca_img.shape[1]
                del data
        except Exception as e:
            logging.error('Error downloading image: {}'.format(e))
            errors_log.append((f.strip(), 'Image Download', str(e)))
            scan_time = timer() - scan_start_time
            continue

        gc.collect()
        image_download_time = timer() - start_time
        logging.info('Image Download Time: {}'.format(image_download_time))

        if ca_img is None:
            print(f'Image not found: {f}')
            continue
        #========== images-to-layouts ==========================================

        start_time = timer()
        # TODO: Don't load the model in with each layout prediction!!
        crops_for_effocr, layout_crops = get_layout_predictions(layout_inf_sess, label_map_layout, ca_img, layout_output, 
                                                            f_idx, layout_input_name, backend = layout_model_backend)
        # try:
        #     crops_for_effocr, layout_crops = get_layout_predictions(layout_inf_sess, label_map_layout, ca_img, layout_output, 
        #                                                         f_idx, layout_input_name, backend = layout_model_backend)
        # except Exception as e:
        #     logging.error('Error getting layout predictions: {}'.format(e))
        #     errors_log.append((f.strip(), 'Layout Prediction', str(e)))
        #     scan_time = timer() - scan_start_time
        #     continue

        gc.collect()

        images_to_layout_time = timer() - start_time
        logging.info(f'Images to Layouts: {images_to_layout_time}')

        metadata = {'page_number': 'na', 'scan_url': f, 'scan_ocr': 'na', 'scan':{}}
        metadata['bboxes'] = []

        for i, (layout_cls, (x0, y0, x1, y1), _) in enumerate(layout_crops):
            bbox_data = {
                'id': i,
                'bbox': {'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1},
                'class': label_map_layout[int(layout_cls.item())],
            }
            
            metadata['bboxes'].append(bbox_data)

    # Save the error log as a csv
    with open(os.path.join(output_save_path, "error_table.csv"), 'w') as outfile:
        outfile.write('filename,stage,error\n')
        outfile.writelines([','.join(error) + '\n' for error in errors_log])
        
    scan_time = timer() - scan_start_time
    print(f'Layout processing took {scan_time} seconds.')
    
    try:
        with open(os.path.join(output_save_path, "img_layout.json"), 'w') as f:
            json.dump(metadata, f)
    except Exception as e:
        print(e)
        print("Could not save JSON for layout output.")

    return None

if __name__ == '__main__':
    print("Start!")
    print('Test push')
    # gc.set_debug(gc.DEBUG_LEAK)

    #========== inputs =============================================

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_save_path",
        help="Path to directory for saving outputs of pipeline inference")
    parser.add_argument("--config_path_layout",
        help="Path to YOLOv5 config file")
    parser.add_argument("--config_path_line",
        help="Path to  YOLOv5 config file")
    parser.add_argument("--checkpoint_path_layout",
        help="Path to Detectron2 compatible checkpoint file (model weights file)")
    parser.add_argument("--checkpoint_path_line",
        help="Path to Detectron2 compatible checkpoint file (model weights file)")
    parser.add_argument("--label_map_path_layout",
        help="Path to JSON file mapping numeric object classes to their labels")
    parser.add_argument("--label_map_path_line",
        help="Path to JSON file mapping numeric object classes to their labels")
    parser.add_argument("--layout-output", default=None,
        help="Path to save layout model images with detections drawn on them")
    parser.add_argument("--manifest_path", default='manifest_0.txt')
    parser.add_argument("--first_n", default=None, type=int)
    parser.add_argument("--layout-line-only", action='store_true', default=False)
    parser.add_argument("--bbox_output", action='store_true', default=False)
    parser.add_argument("--layout_model_backend", type=str, default='yolo')
    parser.add_argument("--punc_padding", type=int, default=0)

    args = parser.parse_args()
    main(args)

    
