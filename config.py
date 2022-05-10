from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
font_path = './fonts/simfang.ttf'
sample_rate = 50000 # sample every [sample_rate] miliseconds

# Parameters
iou_threshold = 0.5
edge_threshold = 0.15
full_video_threshold = 0.7
edit_distance_leeway = 1