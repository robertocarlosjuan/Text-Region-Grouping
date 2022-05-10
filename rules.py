import json
import os
from utils import levenshteinDistance, check_bbox_at_edge, parse_args, processImages, detect_bounding, get_bbox_location_dict, draw_bounding
from config import iou_threshold, edge_threshold, full_video_threshold, edit_distance_leeway

# python rules.py --video_path "/home/hcari/trg/videos/"

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
                    channels.append((json.loads(bbox), pred_texts[0]))
    return channels

def main():
    args = parse_args()
    video_paths = [args.video_path] if os.path.isfile(args.video_path) else [os.path.join(args.video_path,v) for v in os.listdir(args.video_path)] 

    for video_path in video_paths:
        print(video_path)
        results, height, width, output_path = processImages(video_path, detect_bounding, image_dir=args.vis_dir, num_frames_to_save=1)
        print(results)
        bbox_loc, num_frames = get_bbox_location_dict(results, threshold=iou_threshold)
        print("NUM FRAMES: ", num_frames)
        print(bbox_loc)
        channels = find_channel(bbox_loc, num_frames, height, width, edge_threshold=edge_threshold, full_video_threshold=full_video_threshold, edit_leeway=edit_distance_leeway)
        print(channels)
        draw_bounding(output_path, channels, output_dir=None)

main()