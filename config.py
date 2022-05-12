from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
font_path = './fonts/simfang.ttf'
sample_rate = 5000 # sample every [sample_rate] miliseconds

# Parameters
iou_threshold = 0.2
rc_iou=0.5
edge_threshold = (0.35, 0.15)
full_video_threshold = 0.65
edit_sim = 0.8
edit_sim_rolling_caption = 0.1
rc_height_thres = 0.5
hor_leeway=10
