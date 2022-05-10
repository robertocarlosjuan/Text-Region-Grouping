import os
import json
import argparse

import cv2
print(cv2.__version__)
from PIL import Image
from config import ocr, font_path, sample_rate, iou_threshold, edge_threshold, full_video_threshold, edit_distance_leeway
from paddleocr import draw_ocr
from shapely.geometry import Polygon

# python utils.py --video_path "/home/hcari/trg/videos/ChineseSinovacVideo.mp4"

def detect_bounding(img_name):
    # Output will be a list, each item contains bounding box, text and recognition confidence
    # E.g.[[[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]],...]
    ocr_bboxes = ocr.ocr(img_name, cls=True)
    return ocr_bboxes

def draw_bounding(img_path, ocr_bboxes, output_dir):
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

def processImages(video_path, process, image_dir=None, save_frame=False):
    # results will be frame instance, bbox instance, detect bounding box outputs
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    results = []
    height = None
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*sample_rate)) 
        success,image = vidcap.read()
        if height is None:
            height, width, channels = image.shape
        if success:
            result = process(image)
            results.append(result)
            if save_frame:
                output_count = "_%d" % count
                output_name = output_count.join(os.path.splitext(video_path))
                output_path = os.path.join(image_dir,os.path.basename(output_name)) if image_dir is not None else output_name
                cv2.imwrite(output_path, image)     # save frame as JPEG file
        count = count + 1
    
    return results, height, width

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

def check_bbox_at_edge(bbox, height, width, threshold=0.15):
    if type(bbox) == str:
        bbox = json.loads(bbox)
    for edge in bbox:
        is_edge = [False, False]
        if edge[0] < threshold*width or edge[0] > (1-threshold)*width:
            is_edge[0] = True
        if edge[1] < threshold*height or edge[1] > (1-threshold)*height:
            is_edge[1] = True
        if is_edge[0] and is_edge[1]:
            return True
    return False
        


def get_bbox_location_dict(results, threshold=0.5):
    # bbox_location {bbox_coords: [rec1, rec2, ...]}
    bbox_loc = {}
    num_frames = 0
    for frame in results:
        for bbox_info in frame:
            bbox = json.dumps(bbox_info[0])
            pred_text = bbox_info[1][0]
            # Check if bbox is in previous frames
            bbox_in_prev_frames = bbox if bbox in bbox_loc.keys() else None
            if bbox_in_prev_frames is None:
                for key in bbox_loc.keys():
                    iou = get_intersection_over_union(bbox_info[0], json.loads(key))
                    if iou > threshold:
                        bbox_in_prev_frames = key
                        break
            if bbox_in_prev_frames is not None:
                bbox_loc[bbox_in_prev_frames].append(pred_text)
            else:
                bbox_loc[bbox] = [pred_text]
        if len(frame) > 0:
            num_frames += 1
    return bbox_loc, num_frames

def find_channel(bbox_loc, num_frames, height, width, edge_threshold, full_video_threshold=0.7, edit_leeway=3):
    # threshold: proportion of frames that have the bbox
    channels = []
    for bbox, pred_texts in bbox_loc.items():
        # Check if bbox last throughout video
        if len(pred_texts)/num_frames > full_video_threshold:
            # Check if texts are the same
            if all(levenshteinDistance(x, pred_texts[0])<=edit_leeway for x in pred_texts):
                # Check if bbox at edges
                if check_bbox_at_edge(bbox, height, width, edge_threshold):
                    channels.append((json.loads(bbox), pred_texts))
    return channels
                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="path to video")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    video_paths = [args.video_path] if os.path.isfile(args.video_path) else [os.path.join(args.video_path,v) for v in os.listdir(args.video_path)] 

    for video_path in video_paths:
        print(video_path)
        results, height, width = processImages(video_path, detect_bounding)
        print(results)
        bbox_loc, num_frames = get_bbox_location_dict(results, threshold=iou_threshold)
        print("NUM FRAMES: ", num_frames)
        print(bbox_loc)
        channels = find_channel(bbox_loc, num_frames, height, width, edge_threshold=edge_threshold, full_video_threshold=full_video_threshold, edit_leeway=edit_distance_leeway)
        print(channels)

    # print(results)
    # print(bbox_loc)

main()

