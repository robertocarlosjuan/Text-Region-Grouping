from lib2to3.pgen2.literals import simple_escapes
import os
import re
import json
import math
import argparse
import statistics
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from config import *
from paddleocr import draw_ocr
from shapely.geometry import Polygon
import xml.etree.cElementTree as ET
from Levenshtein import distance as lev

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="path to video")
    parser.add_argument("--vis_dir", help="path to save visualisations")
    args = parser.parse_args()
    return args

def get_bbox_location_dict(results, threshold=0.5, rc_iou=0.5):
    # bbox_location {bbox_coords: [(pred_text, conf), ...]}
    bbox_loc = {}
    num_frames = 0
    for frame_no, frame in enumerate(results):
        for bbox_info in frame:
            bbox = json.dumps(bbox_info[0])
            # Check if bbox is in previous frames
            bbox_in_prev_frames = ["iou",bbox] if bbox in bbox_loc.keys() else None
            if bbox_in_prev_frames is None:
                for key in bbox_loc.keys():
                    iou = get_intersection_over_union(bbox_info[0], json.loads(key))
                    if iou > threshold: # Channel, title
                        bbox_in_prev_frames = ["iou", key]
                        break
                if bbox_in_prev_frames is None:
                    for key in bbox_loc.keys():
                        vert_iou = get_vertical_iou(bbox_info[0], json.loads(key)) # Rolling News
                        if vert_iou > rc_iou:
                            union_region = get_union_region(bbox_info[0], json.loads(key))
                            bbox_in_prev_frames = ["vert_iou", json.dumps(union_region)]
                            bbox_loc[bbox_in_prev_frames[1]] = bbox_loc.pop(key)
                            bbox_loc[bbox_in_prev_frames[1]].append((bbox_info[1], json.dumps(bbox_info[0]), frame_no*sample_rate))
                            break

            if bbox_in_prev_frames is not None:
                bbox_loc[bbox_in_prev_frames[1]].append((bbox_info[1], bbox_in_prev_frames[0], frame_no*sample_rate))
            else:
                bbox_loc[bbox] = [(bbox_info[1], "iou", frame_no*sample_rate)]
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
    frames = []
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*sample_rate)) 
        success,image = vidcap.read()
        if height is None:
            height, width, channels = image.shape
        if success:
            result = process(image)
            results.append(result)
            frames.append(image)
            if save_frame < num_frames_to_save:
                output_count = "_%d" % count
                output_name = os.path.splitext(video_path)[0] + output_count + ".jpeg"
                output_path = os.path.join(image_dir,os.path.basename(output_name)) if image_dir is not None else output_name
                cv2.imwrite(output_path, image)     # save frame as JPEG file
                save_frame += 1
        count = count + 1
    
    return results, height, width, frames

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

def overlap(coord1, coord2, type='', x_margin=100, y_margin=10):
    # union_area = Polygon(self.coords).union(Polygon(new_coords)).area
    # # if union_area - Polygon(self.coords).area == 0 or union_area - Polygon(new_coords).area < 10:
    # #     return True
    # iou = Polygon(self.coords).intersection(Polygon(new_coords)).area / union_area
    # return iou > iou_threshold
    mismatch = False
    # coords = [item for sublist in self.coords for item in sublist]
    # new_coords = [item for sublist in new_coords for item in sublist]
    if type not in ['shot', 'no_audio']:
        coord1 = [coord1[0][0], coord1[0][1], coord1[2][0], coord1[2][1]]
        coord2 = [coord2[0][0], coord2[0][1], coord2[2][0], coord2[2][1]]
    for i, (p, c) in enumerate(zip(coord1, coord2)):
        if i%2 == 0:
            margin = x_margin
        else:
            margin = y_margin
        
        if abs(p-c) <= margin:
            continue
        else:
            mismatch = True
            break

    return not mismatch

def height(bbox):
    return bbox[2][1]-bbox[0][1]

def width(bbox):
    return bbox[2][0]-bbox[0][0]

def get_union_region(bbox1, bbox2):
    # area1 = height(bbox1)*width(bbox1)
    # area2 = height(bbox2)*width(bbox2)
    # if area1 > area2:
    #     return bbox1
    # else:
    #     return bbox2
    top_left = (min(bbox1[0][0], bbox2[0][0]),min(bbox1[0][1], bbox2[0][1]))
    top_right = (max(bbox1[1][0], bbox2[1][0]),min(bbox1[1][1], bbox2[1][1]))
    bottom_right = (max(bbox1[2][0], bbox2[2][0]),max(bbox1[2][1], bbox2[2][1]))
    bottom_left = (min(bbox1[3][0], bbox2[3][0]),max(bbox1[3][1], bbox2[3][1]))
    return [top_left, top_right, bottom_right, bottom_left]

def get_horizontal_union_region(bbox1, bbox2):
    top_left = (min(bbox1[0][0], bbox2[0][0]),bbox1[0][1])
    top_right = (max(bbox1[1][0], bbox2[1][0]),bbox1[1][1])
    bottom_right = (max(bbox1[2][0], bbox2[2][0]),bbox1[2][1])
    bottom_left = (min(bbox1[3][0], bbox2[3][0]),bbox1[3][1])
    return [top_left, top_right, bottom_right, bottom_left]

def get_plot_coords(coords):
    x_values = [x[0] for x in coords]
    y_values = [x[1] for x in coords]
    return [min(x_values), min(y_values), max(x_values), max(y_values)]

def get_distance(point1, point2):
    return np.sqrt(np.square(point1 - point2).sum())

def get_centroid(point_list):
    return np.median(point_list, axis=0)

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

def check_rc_by_text(rc):
    rc_sorted = sorted(rc, key=len)
    clusters = {}
    rejected = []
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
                    item_in_clusters = True
                    break
        if not item_in_clusters:
            rejected.append(item)
    refined_clusters = [(k,v) for k, v in clusters.items() if v>1 and v!=max(clusters.values())]
    return len(refined_clusters)>0

def diff_density(pred_texts):
    edit_similarities = []
    for i in range(len(pred_texts)-1):
        edit_similarity = norm_edit_sim(pred_texts[i], pred_texts[i+1])
        edit_similarities.append(edit_similarity)
    if len(edit_similarities) > 0:
        prop_zero = edit_similarities.count(0) / len(edit_similarities)
    else:
        prop_zero = 0
    return prop_zero

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

def check_in(s1, s2):
    if len(s1) >= len(s2):
        return False
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

def check_bbox_at_edge(bbox, height, width, threshold=0.15, mode=None):
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
    if mode!=None:
        return "".join(quad) == mode
    else:
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
    print(edit_similarities)
    avg_edit_sim = sum(edit_similarities)/len(edit_similarities)
    st_dev = statistics.stdev(edit_similarities)
    prop_zero = edit_similarities.count(0) / len(edit_similarities)
    prop_unzero = 1-prop_zero
    print(pred_texts[0][0], avg_edit_sim, st_dev, prop_unzero)
    return edit_similarities, avg_edit_sim, prop_unzero
        
def check_horizontal(bbox, threshold = 10):
    return abs(bbox[0][1] - bbox[1][1]) <= threshold and abs(bbox[2][1] - bbox[3][1]) <= threshold

def reorganise_bbox_for_xml(bbox, text, frame_range):
    print("CHECKING")
    print(bbox)
    print(text)
    print(frame_range)
    text = text[0] if type(text)!= str else text
    x_values = [x[0] for x in bbox]
    y_values = [x[1] for x in bbox]
    new_bbox = [min(x_values), min(y_values), max(x_values), max(y_values)]
    if type(frame_range) == int:
        frame_range = [frame_range, frame_range]
    new_bbox_info = [new_bbox, frame_range, text]
    print(new_bbox_info)
    return new_bbox_info

def generate_xml(bboxes_all_types, types, base_filename, output_path):
    # bbox format [[topleftx, toplefty,botrightx, botrighty], [frame_start, ..., frame_end], text]
    xml_string = ['<detection_list>\n']
    for bboxes, bbox_type in zip(bboxes_all_types, types):
        if len(bboxes)==0:
            continue
        xml_string.append('\t<detection type={}>\n'.format(bbox_type))
        for bbox in bboxes:
            print(bbox)
            xml_string.append('\t\t<frame start={} end={}>\n'.format(bbox[1][0], bbox[1][-1]))
            xml_string.append('\t\t\t<bbox>\n\t\t\t\t<topleft>\n\t\t\t\t\t<x>{}</x>\n\t\t\t\t\t<y>{}</y>\n\t\t\t\t</topleft>\n'.format(bbox[0][0], bbox[0][1]))
            xml_string.append('\t\t\t\t<botright>\n\t\t\t\t\t<x>{}</x>\n\t\t\t\t\t<y>{}</y>\n\t\t\t\t</botright>\n\t\t\t</bbox>\n'.format(bbox[0][2], bbox[0][-1]))
            xml_string.append('\t\t\t<text>{}</text>\n<\t\t</frame>\n'.format(bbox[-1]))
        xml_string.append('\t</detection>\n')
    xml_string.append('</detection_list>')
    xml_string = ''.join(xml_string)
    xml_filename = "{}/{}.xml".format(output_path, base_filename)
    xml_file = open(xml_filename, "w")
    xml_file.write(xml_string)
    xml_file.close()
    print('Saved to {}'.format(xml_filename))

def plot_bbox(img, label_name, bbox, colour=(0, 255, 0)):
    (w, h), _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[-1])), colour, 2)
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1]) - 20), (int(bbox[0]) + w, int(bbox[1])), colour, -1)
    img = cv2.putText(img, label_name, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img

def save_to_video(frames, bboxes_all_types, types, video_path, out_dir, base_filename):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #width of image
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(1/sample_rate)
    fps = 1 if fps < 1 else fps
    out_file = '{}/{}.mp4'.format(out_dir, base_filename)
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    # Plot bboxes and output video
    frames_dict = {}
    for idx, frame in enumerate(frames):
        # img = cv2.imread(frame)
        frames_dict[idx*sample_rate] = frame

    for type, bboxes in zip(types, bboxes_all_types):
        for bbox in bboxes:
            for frame_no in bbox[1]:
                frames_dict[frame_no] = plot_bbox(frames_dict[frame_no], type+": "+bbox[-1], bbox[0], colours[type])

    for i in range(idx):
        out.write(frames_dict[i*sample_rate])
    out.release()
    cap.release()

def load_known_channels(known_channels_path):
    with open(known_channels_path) as f:
        known_channels = f.read().splitlines()
    return known_channels

def save_known_channels(known_channels, known_channels_path):
    with open(known_channels_path, "w") as f:
        f.write("\n".join(known_channels))

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def track_bboxes(frames, type='', x_margin=100, y_margin=10):
    bbox_dict = {}
    bbox_hist = []
    feature_frame_idx = []
    for idx, frame in tqdm(enumerate(frames)):
        if type != 'no_audio':
            ocr_bboxes = ocr.ocr(frame, cls=True)
        else:
            ocr_bboxes = frame
        if type not in ['shot', 'no_audio']:
            classifier_results = classify(frame)
            if classifier_results[0]['label'] == 'Feature frame':
                feature_frame_idx.append(idx)

        coors_to_delete = []
        for tracked_coor, value in bbox_dict.items():
            frame_range = value[0]
            tracked_text = value[-1][0]
            continued = False
            for curr_bbox in ocr_bboxes:
                curr_text = curr_bbox[-1][0]
                curr_coor = tuple([curr_bbox[0][0][0], curr_bbox[0][0][1], curr_bbox[0][2][0], curr_bbox[0][2][1]])
                if overlap(tracked_coor, curr_coor, type, x_margin, y_margin) and idx == frame_range[-1] + 1 and lev(curr_text, tracked_text) < 10:
                    bbox_dict[tracked_coor][0][-1] = idx
                    continued = True
                    break
            if not continued:
                coors_to_delete.append(tracked_coor)
                if frame_range[-1] - frame_range[0] >= 0:
                    bbox_hist.append([tracked_coor, frame_range, tracked_text])
        
        for curr_bbox in ocr_bboxes:
            curr_text = curr_bbox[-1]
            curr_coor = tuple([curr_bbox[0][0][0], curr_bbox[0][0][1], curr_bbox[0][2][0], curr_bbox[0][2][1]])
            already_tracked = False
            for tracked_coor, frame_range in bbox_dict.items():
                if overlap(tracked_coor, curr_coor, type, x_margin, y_margin):
                    already_tracked = True
                    break
            if not already_tracked:
                bbox_dict[curr_coor] = [[idx, idx], curr_text]

        for coor in coors_to_delete:
            del bbox_dict[coor]

    return bbox_hist, feature_frame_idx

def find_subtitle_bbox(subtitle_bboxes, num_frames, bbox, height):
    # Condition for detecting subtitle bbox
    in_ms = num_frames * sample_rate
    if in_ms >= 0 and in_ms <= 4000 and bbox[0][-1] >= int(height/2):
        subtitle_bboxes.append(bbox)

def find_scene_text(shot_bbox_hist, shot_info):
    scene_text_bboxes = []
    for bbox in shot_bbox_hist:
        frame_range = bbox[1]
        num_frames = frame_range[-1] - frame_range[0]
        # Detect scene text
        if num_frames > 0:
            continue
        else:
            shot_row = shot_info.loc[shot_info[0]==frame_range[0]]
            start_frame = list(shot_row[1])[0]
            end_frame = list(shot_row[2])[0]
            scene_text_bboxes.append([bbox[0], [start_frame, end_frame], bbox[-1]])

    return scene_text_bboxes

def translate_frame(fps, frame_no):
    frame_gap = int(sample_rate/fps)
    
    return frame_gap*frame_no

# def shortened_norm_edit_sim(s1, s2):
#     s1 = s1[0] if type(s1)==tuple else s1
#     s2 = s2[0] if type(s2)==tuple else s2
#     shorter = len(s1) if len(s1) < len(s2) else len(s2)
#     cut_off = math.ceil(shorter*0.8)
#     return norm_edit_sim(s1[cut_off*-1:], s2[:cut_off])

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