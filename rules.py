import json
import os
from utils import *
from config import *

# python rules.py --video_path "/home/hcari/trg/videos/" --vis_dir /home/hcari/trg/vis/

def process(bbox_loc, num_frames, height, width, edge_threshold, full_video_threshold=0.7, edit_sim=0.7, edit_sim_rolling_caption=0.5, hor_leeway=1, rc_height_thres=0.15):
    # threshold: proportion of frames that have the bbox
    channels = []
    titles = []
    rolling_captions = []
    scene_texts = []
    for bbox, pred_dict in bbox_loc.items():
        pred_texts = []
        rc_pred_texts = []
        combined = []
        static_frame_range = []
        frame_range = []
        not_rc_bboxes = {}
        for x in pred_dict:
            if x[1] not in ["iou", "vert_iou"]:
                if bbox in not_rc_bboxes.keys():
                    not_rc_bboxes[bbox].append(x)
                else:
                    not_rc_bboxes[bbox] = [x]
                continue
            if x[1]=="iou":
                pred_texts.append(x[0])
                static_frame_range.append(x[2])
            if x[1]=="vert_iou":
                rc_pred_texts.append(x[0])
            combined.append(x[0])
            frame_range.append(x[2])
        print(combined[0])
        print(bbox)
        # Check bbox_horizontal
        if check_horizontal(json.loads(bbox), threshold = hor_leeway):
            print("is_horizontal")
            # Check if bbox last throughout video
            if len(combined)/num_frames > full_video_threshold:
                print("combined last through video")
                edit_similarities, avg_edit_sim, prop_unzero = get_edit_similarities(combined)
                print(edit_similarities)
                print("avg_edit_sim: ", avg_edit_sim)
                print("prop_unzero: ", prop_unzero)
                if len(pred_texts)/num_frames > full_video_threshold:
                    sedit_similarities, savg_edit_sim, sprop_unzero = get_edit_similarities(pred_texts)
                    print("pred_text last through video")
                    # Check if texts are the same
                    if sprop_unzero>edit_sim: #avg_edit_sim >= edit_sim
                        print("Same text across frames")
                        # Check if bbox at edges
                        if check_bbox_at_edge(bbox, height, width, edge_threshold):
                            print("Bounding box at the edge")
                            channels.append(reorganise_bbox_for_xml(json.loads(bbox), pred_texts[0], static_frame_range))
                            print("channels\n")
                        else:
                            print("Bounding box not at the edge")
                            titles.append(reorganise_bbox_for_xml(json.loads(bbox), pred_texts[0], static_frame_range))
                            print("titles\n")
                    else:
                        # Check if rolling captions
                        # Check if text in next frame similar to this frame
                        # Check that bbox is in bottom area of frame
                        print("Texts are not always the same")
                        below = check_bbox_below(bbox, height, threshold = rc_height_thres)
                        is_rc_text = maybe_is_rc(rc_pred_texts)
                        print("rc_pred_text: ", rc_pred_texts)
                        print("Below: ", below)
                        print("is_rc_text: ", is_rc_text)
                        if below and is_rc_text:
                            rolling_captions.append(reorganise_bbox_for_xml(json.loads(bbox), rc_pred_texts[0], frame_range))
                            print("rolling caption\n")
                        else:
                            for not_rc_bbox in not_rc_bboxes[bbox]:
                                scene_texts.append(reorganise_bbox_for_xml(json.loads(not_rc_bbox[1]), not_rc_bbox[0], not_rc_bbox[2]))
                            print("scene_text\n")
                else:
                    if prop_unzero>edit_sim: #avg_edit_sim >= edit_sim
                        titles.append(reorganise_bbox_for_xml(json.loads(bbox), pred_texts[0], static_frame_range))
                        print("titles\n")
                    else:
                        print("mostly vert_iou: ", len(rc_pred_texts)/num_frames > full_video_threshold)
                        below = check_bbox_below(bbox, height, threshold = rc_height_thres)
                        is_rc_text = maybe_is_rc(rc_pred_texts)
                        print("rc_pred_text: ", rc_pred_texts)
                        print("Below: ", below)
                        print("is_rc_text: ", is_rc_text)
                        if below and is_rc_text:
                            rolling_captions.append(reorganise_bbox_for_xml(json.loads(bbox), rc_pred_texts[0], frame_range))
                            print("rolling caption\n")
                        elif len(rc_pred_texts) < len(pred_texts):
                            for not_rc_bbox in not_rc_bboxes[bbox]:
                                scene_texts.append(reorganise_bbox_for_xml(json.loads(not_rc_bbox[1]), not_rc_bbox[0], not_rc_bbox[2]))
                            print("scene_text\n")
            else:
                scene_texts.append(reorganise_bbox_for_xml(json.loads(bbox), combined[0], frame_range))
                print("scene_text\n")
        else:
            print("not horizontal")
            scene_texts.append(reorganise_bbox_for_xml(json.loads(bbox), combined[0], frame_range))
            print("scene_text\n")
    return channels, titles, rolling_captions, scene_texts

def main():
    args = parse_args()
    video_paths = [args.video_path] if os.path.isfile(args.video_path) else [os.path.join(args.video_path,v) for v in os.listdir(args.video_path)] 

    for video_path in video_paths:
        print(video_path)
        results, height, width, frames = processImages(video_path, detect_bounding, image_dir=args.vis_dir, num_frames_to_save=1)
        bbox_loc, num_frames = get_bbox_location_dict(results, threshold=iou_threshold, rc_iou=rc_iou)
        print("NUM FRAMES: ", num_frames)
        channels, titles, rolling_captions, scene_texts = process(
                                                            bbox_loc, 
                                                            num_frames, 
                                                            height, width, 
                                                            edge_threshold=edge_threshold, 
                                                            full_video_threshold=full_video_threshold, 
                                                            edit_sim=edit_sim, 
                                                            edit_sim_rolling_caption=edit_sim_rolling_caption,
                                                            hor_leeway=hor_leeway, 
                                                            rc_height_thres=rc_height_thres)
        bboxes_all_types = [channels, titles, rolling_captions, scene_texts]
        types = ["channels", "titles", "rolling_captions", "scene_texts"]
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        generate_xml(bboxes_all_types, types, base_filename, args.vis_dir)
        save_to_video(frames, bboxes_all_types, types, video_path, args.vis_dir, base_filename+"_results")
        # draw_bounding(output_path, channels, output_dir=args.vis_dir, label="_channels")
        # draw_bounding(output_path, titles, output_dir=args.vis_dir, label="_titles")
        # draw_bounding(output_path, rolling_captions, output_dir=args.vis_dir, label="_roling_captions")
        # draw_bounding(output_path, scene_texts, output_dir=args.vis_dir, label="_scene_texts")

main()