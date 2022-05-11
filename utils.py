import os
import json
import math
import argparse
import numpy as np
import cv2
print(cv2.__version__)
from PIL import Image
from config import ocr, font_path, sample_rate
from paddleocr import draw_ocr
from shapely.geometry import Polygon

# python utils.py --video_path "/home/hcari/trg/videos/ChineseSinovacVideo.mp4"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="path to video")
    parser.add_argument("--vis_dir", help="path to save visualisations")
    args = parser.parse_args()
    return args

def get_bbox_location_dict(results, threshold=0.5):
    # bbox_location {bbox_coords: [rec1, rec2, ...]}
    bbox_loc = {}
    num_frames = 0
    for frame in results:
        for bbox_info in frame:
            bbox = json.dumps(bbox_info[0])
            # Check if bbox is in previous frames
            bbox_in_prev_frames = bbox if bbox in bbox_loc.keys() else None
            if bbox_in_prev_frames is None:
                for key in bbox_loc.keys():
                    iou = get_intersection_over_union(bbox_info[0], json.loads(key))
                    if iou > threshold:
                        bbox_in_prev_frames = key
                        break
            if bbox_in_prev_frames is not None:
                bbox_loc[bbox_in_prev_frames].append(bbox_info[1])
            else:
                bbox_loc[bbox] = [bbox_info[1]]
        if len(frame) > 0:
            num_frames += 1
    return bbox_loc, num_frames    

def processImages(video_path, process, image_dir=None, num_frames_to_save = 1):
    # results will be frame instance, bbox instance, detect bounding box outputs
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    results = []
    output_path = None
    height = None
    save_frame = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*sample_rate)) 
        success,image = vidcap.read()
        if height is None:
            height, width, channels = image.shape
        if success:
            result = process(image)
            results.append(result)
            if save_frame < num_frames_to_save:
                output_count = "_%d" % count
                output_name = os.path.splitext(video_path)[0] + output_count + ".jpeg"
                output_path = os.path.join(image_dir,os.path.basename(output_name)) if image_dir is not None else output_name
                cv2.imwrite(output_path, image)     # save frame as JPEG file
                save_frame += 1
        count = count + 1
    
    return results, height, width, output_path

# PaddleOCR
def detect_bounding(img_name):
    # Output will be a list, each item contains bounding box, text and recognition confidence
    # E.g.[[[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]],...]
    ocr_bboxes = ocr.ocr(img_name, cls=True)
    return ocr_bboxes

def draw_bounding(img_path, ocr_bboxes, output_dir=None):
    # draw bouding boxes on image
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in ocr_bboxes]
    txts = [line[1][0] for line in ocr_bboxes]
    scores = [line[1][1] for line in ocr_bboxes]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    output_path = "_bbox".join(os.path.splitext(img_path))
    output_path = output_path if output_dir is None else os.path.join(output_dir, os.path.basename(output_path))
    im_show.save(output_path)

# Find Channel
def get_intersection_over_union(pD, pG):
    return Polygon(pD).intersection(Polygon(pG)).area / Polygon(pD).union(Polygon(pG)).area

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# def check_bbox_at_edge(bbox, height, width, threshold=0.15):
#     if type(bbox) == str:
#         bbox = json.loads(bbox)
#     for edge in bbox:
#         is_edge = [False, False]
#         if edge[0] < threshold*width or edge[0] > (1-threshold)*width:
#             is_edge[0] = True
#         if edge[1] < threshold*height or edge[1] > (1-threshold)*height:
#             is_edge[1] = True
#         if not (is_edge[0] and is_edge[1]):
#             return False
#     return True      

def check_bbox_at_edge(bbox, height, width, threshold=0.15):
    print("HEIGHT WIDTH", height, width)
    if type(bbox) == str:
        bbox = json.loads(bbox)
    quad = [None,None]
    find_quad = True
    for edge in bbox:
        edge_quad = [None, None]
        if edge[0] < threshold[0]*width:
            if find_quad:
                quad[0] = "left"
            elif quad[0] != "left":
                return False
            edge_quad[0] = "left"
        if edge[0] > (1-threshold[0])*width:
            if find_quad:
                quad[0] = "right"
            elif quad[0] != "right":
                return False
            edge_quad[0] = "right"
        if edge[1] < threshold[1]*height:
            if find_quad:
                quad[1] = "top"
            elif quad[1] != "top":
                return False
            edge_quad[1] = "top"
        if edge[1] > (1-threshold[1])*height:
            if find_quad:
                quad[1] = "bottom"
            elif quad[1] != "bottom":
                return False
            edge_quad[1] = "bottom"
        if quad[0] is None or quad[1] is None:
            return False
        if find_quad:
            print("QUAD", quad)
        if edge_quad[0] is None or edge_quad[1] is None:
            return False
        print("EDGE", edge_quad)
        find_quad = False
    print(bbox)
    return True
        





