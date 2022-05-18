from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
font_path = './fonts/simfang.ttf'
sample_rate = 5000 # sample every [sample_rate] miliseconds

# Parameters
iou_threshold = 0.2
vertical_iou_thres  = 0.5
rc_iou=0.5
prop_zero_thres = 0.5
prop_zero_group_thres = 0.1
edge_threshold = (0.35, 0.15)
full_video_threshold = 0.65
edit_sim = 0.8
edit_sim_rolling_caption = 0.1
rc_height_thres = 0.5
hor_leeway=10
color_dist_thres = 50
min_length_texts_for_rc = 400
temporal_dist_thres = 5*sample_rate
prop_bbox_video_temporally_thres = 0.8
corner_threshold=(0.35, 0.15)
channel_edit_sim = 0.8
lower_height_threshold = 0.333
colour_density_epsilon = 10
min_colour_density = 0.4
min_avg_length_texts_topicrc = 10


# conda install -c pytorch faiss-gpu

known_channels = ["thestraitstimes", "cna"]

colours = {"channels": (0, 255, 0), #green
            "titles": (255, 0, 0), # red
            "rolling_captions": (0, 0, 255), # blue
            "scene_texts": (255, 255, 0)} # yellow
