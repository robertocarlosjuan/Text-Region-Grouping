import os
import json
import argparse

import cv2
print(cv2.__version__)
from PIL import Image
from config import ocr, font_path, sample_rate
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
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*sample_rate)) 
        success,image = vidcap.read()
        if success:
            result = process(image)
            results.append(result)
            if save_frame:
                output_count = "_%d" % count
                output_name = output_count.join(os.path.splitext(video_path))
                output_path = os.path.join(image_dir,os.path.basename(output_name)) if image_dir is not None else output_name
                cv2.imwrite(output_path, image)     # save frame as JPEG file
        count = count + 1
    return results

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

def get_bbox_location_dict(results, threshold=0.98):
    # bbox_location {bbox_coords: [rec1, rec2, ...]}
    bbox_loc = {}
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
    return bbox_loc
                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="path to video")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    results = processImages(args.video_path, detect_bounding)
    bbox_loc = get_bbox_location_dict(results)

    # print(results)
    print(bbox_loc)

main()

# bbox_loc

# {'[[79.0, 31.0], [161.0, 31.0], [161.0, 72.0], [79.0, 72.0]]': ['抖音'], 
# '[[18.0, 88.0], [273.0, 94.0], [272.0, 122.0], [17.0, 116.0]]': ['抖音号：paitoudangan'], 
# '[[20.0, 288.0], [700.0, 288.0], [700.0, 337.0], [20.0, 337.0]]': ['中国疫苗在新加坡受追捧', '中国疫苗在新加坡受追捧'], 
# '[[30.0, 462.0], [178.0, 466.0], [177.0, 503.0], [29.0, 500.0]]': ['绝对军视'], 
# '[[306.0, 645.0], [624.0, 566.0], [641.0, 637.0], [323.0, 716.0]]': ['SINOVAC'], 
# '[[312.0, 713.0], [644.0, 657.0], [650.0, 693.0], [318.0, 748.0]]': ['科兴控股生物技术有限公司'], 
# '[[315.0, 737.0], [648.0, 700.0], [651.0, 732.0], [319.0, 769.0]]': ['SINGVAC BIOTECH LTD'], 
# '[[24.0, 457.0], [178.0, 463.0], [176.0, 504.0], [22.0, 498.0]]': ['绝对军视'], 
# '[[328.0, 797.0], [373.0, 797.0], [373.0, 811.0], [328.0, 811.0]]': ['即就'], 
# '[[564.0, 1168.0], [699.0, 1168.0], [699.0, 1220.0], [564.0, 1220.0]]': ['小抖音'], 
# '[[446.0, 1232.0], [703.0, 1235.0], [703.0, 1261.0], [446.0, 1259.0]]': ['抖音号：paitoudangan']}

