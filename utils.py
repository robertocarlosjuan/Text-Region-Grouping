from lib2to3.pgen2.literals import simple_escapes
import os
import json
import math
import argparse
import statistics
import numpy as np
import cv2
from PIL import Image
from config import ocr, font_path, sample_rate
from paddleocr import draw_ocr
from shapely.geometry import Polygon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="path to video")
    parser.add_argument("--vis_dir", help="path to save visualisations")
    args = parser.parse_args()
    return args

def get_bbox_location_dict(results, threshold=0.5, rc_iou=0.5):
    # bbox_location {bbox_coords: [rec1, rec2, ...]}
    bbox_loc = {}
    num_frames = 0
    for frame in results:
        for bbox_info in frame:
            bbox = json.dumps(bbox_info[0])
            # Check if bbox is in previous frames
            bbox_in_prev_frames = ("iou",bbox) if bbox in bbox_loc.keys() else None
            if bbox_in_prev_frames is None:
                for key in bbox_loc.keys():
                    iou = get_intersection_over_union(bbox_info[0], json.loads(key))
                    if iou > threshold: # Channel, title
                        bbox_in_prev_frames = ("iou", key)
                        break
                if bbox_in_prev_frames is None:
                    for key in bbox_loc.keys():
                        vert_iou = get_vertical_iou(bbox_info[0], json.loads(key)) # Rolling Captions
                        if vert_iou > rc_iou:
                            bbox_in_prev_frames = ("vert_iou", key)
                            break

            if bbox_in_prev_frames is not None:
                bbox_loc[bbox_in_prev_frames[1]].append((bbox_info[1], bbox_in_prev_frames[0]))
            else:
                bbox_loc[bbox] = [(bbox_info[1], "iou")]
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

def draw_bounding(img_path, ocr_bboxes, output_dir=None, label="_bbox"):
    # draw bouding boxes on image
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in ocr_bboxes]
    txts = [line[1][0] for line in ocr_bboxes]
    scores = [line[1][1] for line in ocr_bboxes]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    output_path = label.join(os.path.splitext(img_path))
    if output_dir is None:
        output_path = output_path  
    elif os.path.isdir(output_dir):
        output_path = os.path.join(output_dir, os.path.basename(output_path))
    elif output_dir.endswith(".jpg") or output_dir.endswith(".jpeg"):
        output_path = output_dir
    im_show.save(output_path)

# Find Channel
def get_vertical_iou(pD, pG):
    min_pD = min([x[1] for x in pD])
    max_pD = max([x[1] for x in pD])
    min_pG = min([x[1] for x in pG])
    max_pG = max([x[1] for x in pG])
    min_union = min_pD if min_pD < min_pG else min_pG
    min_inter = min_pG if min_pD < min_pG else min_pD
    max_union = max_pD if max_pD > max_pG else max_pG
    max_inter = max_pG if max_pD > max_pG else max_pD
    if max_inter < min_inter:
        return 0
    else:
        return (max_inter-min_inter) / (max_union-min_union)

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
    return distances

def norm_edit_sim(s1, s2):
    s1 = s1[0] if type(s1)==tuple else s1
    s2 = s2[0] if type(s2)==tuple else s2
    m = len(s1) if len(s1) < len(s2) else len(s2)
    d = levenshteinDistance(s1, s2)[-1]
    if d>=m:
        return 0
    else:
        return 1.0 / math.exp( d / (m - d) )

def shortened_norm_edit_sim(s1, s2):
    s1 = s1[0] if type(s1)==tuple else s1
    s2 = s2[0] if type(s2)==tuple else s2
    shorter = len(s1) if len(s1) < len(s2) else len(s2)
    cut_off = math.ceil(shorter*0.8)
    return norm_edit_sim(s1[cut_off*-1:], s2[:cut_off])

# def maybe_is_rc(pred_texts, rc_text_thres=0.5):
#     is_rc = 0
#     not_rc = 0
#     for i in range(len(pred_texts)-1):
#         sim = norm_edit_sim(pred_texts[i][0], pred_texts[i+1][0])
#         half_sim = shortened_norm_edit_sim(pred_texts[i][0], pred_texts[i+1][0])
#         if half_sim > sim:
#             is_rc += 1
#         else:
#             not_rc += 1
#     norm_rc_likelihood = is_rc / (is_rc + not_rc)
#     return True if norm_rc_likelihood > rc_text_thres else False

def check_in(s1, s2):
    s2 = s2[:len(s1)]
    return norm_edit_sim(s1, s2) > 0.5

# Check for clusters of text
def maybe_is_rc(rc):
    rc_sorted = sorted([x[0] for x in rc], key=len)
    clusters = {}
    for item in rc_sorted:
        item_in_clusters = False
        for key in clusters.keys():
            if check_in(item, key):
                clusters[key] += 1
                item_in_clusters = True
                break
        if not item_in_clusters:
            for i in range(len(rc_sorted)-1,0,-1):
                key = rc_sorted[i]
                if check_in(item, key):
                    clusters[key] = 1
                    break
    refined_clusters = [(k,v) for k, v in clusters.items() if v>1 and v!=max(clusters.values())]
    return len(refined_clusters)>0

def check_bbox_below(bbox, height, threshold=0.15):
    if type(bbox) == str:
        bbox = json.loads(bbox)
    if all(edge[1] < threshold*height for edge in bbox) or all(edge[1] > (1-threshold)*height for edge in bbox):
        return True
    else:
        return False

def check_bbox_at_edge(bbox, height, width, threshold=0.15):
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
        if edge_quad[0] is None or edge_quad[1] is None:
            return False
        find_quad = False
    return True

def reject_outliers(data, m=5):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return np.squeeze(data[s<m]).tolist()

def get_edit_similarities(pred_texts):
    edit_similarities = []
    for i in range(len(pred_texts)-1):
        edit_similarity = norm_edit_sim(pred_texts[i][0], pred_texts[i+1][0])
        edit_similarities.append(edit_similarity)
    
    edit_similarities = reject_outliers(edit_similarities)
    # print(edit_similarities)
    avg_edit_sim = sum(edit_similarities)/len(edit_similarities)
    st_dev = statistics.stdev(edit_similarities)
    prop_zero = edit_similarities.count(0) / len(edit_similarities)
    prop_unzero = 1-prop_zero
    print(pred_texts[0][0], avg_edit_sim, st_dev, prop_unzero)
    return edit_similarities, avg_edit_sim, prop_unzero
        
def check_horizontal(bbox, threshold = 10):
    return abs(bbox[0][1] - bbox[1][1]) <= threshold and abs(bbox[2][1] - bbox[3][1]) <= threshold




