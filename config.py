from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
font_path = './fonts/simfang.ttf'
sample_rate = 50000 # sample every [sample_rate] miliseconds