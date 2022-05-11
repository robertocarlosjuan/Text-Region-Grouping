import os
import re
import cv2
import argparse
from tqdm import tqdm
from glob import glob

import speech_recognition as sr
from config import ocr
from img_classifier.CLIP.demo import classify

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def check_coor_match(prev_coor, curr_coor, margin):
    mismatch = False
    for p, c in zip(prev_coor, curr_coor):
        if abs(p-c) <= margin:
            continue
        else:
            mismatch = True
            break

    return mismatch

def check_if_feature_frame(frame_range, feature_frame_idx):
    is_feature_frame = True
    for i in range(frame_range[0], frame_range[-1]+1):
        if i not in feature_frame_idx:
            is_feature_frame = False
            break
    return is_feature_frame

def remove_duplicate_bbox(frame_to_subtitle, frame_to_name):
    for (k_sub, v_sub), (k_name, v_name) in zip(frame_to_subtitle.items(), frame_to_name.items()):
        new_v_name = []
        if k_sub == k_name:
            for bbox_name in v_name:
                remove_coor = False
                for bbox_sub in v_sub:
                    if not check_coor_match(bbox_sub, bbox_name, 5):
                        remove_coor = True
                        break
                if not remove_coor:
                    new_v_name.append(bbox_name)
        frame_to_name[k_name] = new_v_name
    
    return frame_to_name

def split_vid_to_frames(cap, frame_save_path, base_filename):
    success,image = cap.read()
    count = 0
    while success:
        cv2.imwrite("{}/{}_{}.jpg".format(frame_save_path, base_filename, count), image)     # save frame as JPEG file      
        success, image = cap.read()
        # print('Read a new frame: ', success)
        count += 1

def track_bboxes(frames):
    bbox_dict = {}
    bbox_hist = []
    feature_frame_idx = []
    for idx, frame in tqdm(enumerate(frames[750:1000])):
        ocr_bboxes = ocr.ocr(frame, cls=True)
        classifier_results = classify(frame)
        if classifier_results[0]['label'] == 'Feature frame':
            feature_frame_idx.append(idx)

        coors_to_delete = []
        for tracked_coor, frame_range in bbox_dict.items():
            continued = False
            for curr_bbox in ocr_bboxes:
                curr_coor = tuple([curr_bbox[0][0][0], curr_bbox[0][0][1], curr_bbox[0][2][0], curr_bbox[0][2][1]])
                if not check_coor_match(tracked_coor, curr_coor, 5) and idx == frame_range[-1] + 1:
                    bbox_dict[tracked_coor][-1] = idx
                    continued = True
                    break
            if not continued:
                coors_to_delete.append(tracked_coor)
                if frame_range[-1] - frame_range[0] > 0:
                    bbox_hist.append([tracked_coor, frame_range])
        
        for curr_bbox in ocr_bboxes:
            curr_coor = tuple([curr_bbox[0][0][0], curr_bbox[0][0][1], curr_bbox[0][2][0], curr_bbox[0][2][1]])
            already_tracked = False
            for tracked_coor, frame_range in bbox_dict.items():
                if not check_coor_match(tracked_coor, curr_coor, 5):
                    already_tracked = True
                    break
            if not already_tracked:
                bbox_dict[curr_coor] = [idx, idx]

        for coor in coors_to_delete:
            del bbox_dict[coor]

    return bbox_hist, feature_frame_idx

def find_subtitle_bbox(frame_to_subtitle, frame_range, num_frames, bbox, fps, height):
    # Condition for detecting subtitle bbox
    if num_frames >= 10 and num_frames <= fps*3 and bbox[0][-1] >= int(height/2):
        for i in range(frame_range[0], frame_range[-1] + 1):
            frame_to_subtitle[i].append(bbox[0])

    return frame_to_subtitle

def find_name_bbox(frame_to_name, frame_range, num_frames, bbox, fps, height, feature_frame_idx):
    # Condition for detecting name bbox
    if check_if_feature_frame(frame_range, feature_frame_idx) and num_frames >= 5 and num_frames <= fps*10 and bbox[0][-1] >= int(height/2):
        for i in range(frame_range[0], frame_range[-1] + 1):
            frame_to_name[i].append(bbox[0])
    
    return frame_to_name

def plot_bbox(img, label_name, bbox, colour=(0, 255, 0)):
    (w, h), _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[-1])), colour, 2)
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1]) - 20), (int(bbox[0]) + w, int(bbox[1])), colour, -1)
    img = cv2.putText(img, label_name, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img

def main(args):
    lang = args.lang
    input_path = '../montage_detection/sinovac_ver/ref_video' 
    frame_save_path = '../non_commit_trg/frames'
    output_path = '../non_commit_trg/temp'
    os.makedirs(frame_save_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    base_filename = 'G8DOaR7tNAs'
    cap = cv2.VideoCapture('{}/{}.mp4'.format(input_path, base_filename))
    fps = round(cap.get(cv2.CAP_PROP_FPS)) 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #width of image
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_file = '{}/{}.mp4'.format(output_path, base_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    # Split video into frames
    # split_vid_to_frames(cap, frame_save_path, base_filename)

    # Track bboxes across frames
    frames = glob('{}/{}_*.jpg'.format(frame_save_path, base_filename))
    frames.sort(key=natural_keys)
    # bbox_hist, feature_frame_idx = track_bboxes(frames)

    # bbox_hist = [[(313.0,712.0,650.0,693.0),[0,2]],[(315.0,737.0,651.0,731.0),[0,4]],[(25.0,24.0,163.0,75.0),[2,5]],[(307.0,645.0,642.0,636.0),[0,8]],[(299.0,787.0,456.0,813.0),[7,8]],[(349.0,704.0,652.0,692.0),[3,10]],[(308.0,642.0,648.0,632.0),[9,10]],[(73.0,29.0,164.0,73.0),[8,11]],[(299.0,787.0,458.0,813.0),[10,11]],[(25.0,24.0,163.0,75.0),[12,15]],[(308.0,648.0,647.0,629.0),[11,17]],[(358.0,702.0,656.0,689.0),[11,18]],[(299.0,787.0,452.0,813.0),[12,19]],[(73.0,29.0,164.0,73.0),[18,19]],[(306.0,640.0,651.0,627.0),[18,19]],[(299.0,787.0,458.0,813.0),[20,21]],[(353.0,700.0,662.0,687.0),[19,23]],[(306.0,640.0,656.0,626.0),[21,23]],[(299.0,787.0,452.0,813.0),[22,24]],[(25.0,24.0,163.0,75.0),[22,25]],[(348.0,702.0,669.0,683.0),[24,28]],[(306.0,636.0,662.0,625.0),[24,29]],[(299.0,787.0,458.0,813.0),[25,29]],[(73.0,29.0,164.0,73.0),[28,31]],[(342.0,702.0,673.0,682.0),[29,31]],[(332.0,705.0,677.0,681.0),[32,33]],[(25.0,23.0,163.0,77.0),[32,35]],[(299.0,787.0,452.0,813.0),[30,37]],[(326.0,705.0,677.0,679.0),[34,39]],[(73.0,29.0,164.0,73.0),[38,39]],[(305.0,639.0,667.0,619.0),[30,40]],[(308.0,636.0,670.0,613.0),[41,43]],[(25.0,24.0,163.0,75.0),[42,43]],[(130.0,789.0,588.0,812.0),[42,45]],[(73.0,29.0,164.0,73.0),[48,49]],[(25.0,28.0,160.0,75.0),[53,55]],[(73.0,29.0,164.0,73.0),[58,59]],[(25.0,23.0,163.0,77.0),[62,65]],[(30.0,462.0,177.0,503.0),[0,69]],[(73.0,29.0,164.0,73.0),[68,69]],[(182.0,648.0,285.0,668.0),[70,72]],[(343.0,665.0,667.0,681.0),[70,72]],[(495.0,825.0,602.0,843.0),[71,72]],[(177.0,505.0,504.0,524.0),[70,73]],[(339.0,815.0,428.0,829.0),[70,73]],[(30.0,465.0,189.0,500.0),[71,74]],[(25.0,28.0,160.0,75.0),[73,75]],[(490.0,821.0,713.0,843.0),[73,75]],[(336.0,663.0,667.0,681.0),[73,76]],[(28.0,465.0,180.0,502.0),[75,76]],[(171.0,504.0,502.0,524.0),[74,77]],[(179.0,647.0,279.0,669.0),[74,77]],[(330.0,815.0,422.0,828.0),[74,78]],[(334.0,592.0,385.0,605.0),[76,78]],[(27.0,466.0,173.0,496.0),[77,78]],[(26.0,493.0,97.0,507.0),[77,79]],[(331.0,663.0,661.0,681.0),[77,79]],[(73.0,29.0,164.0,73.0),[78,79]],[(483.0,821.0,716.0,841.0),[76,80]],[(27.0,465.0,162.0,498.0),[79,80]],[(553.0,600.0,584.0,609.0),[79,80]],[(585.0,759.0,615.0,768.0),[80,81]],[(165.0,505.0,496.0,524.0),[78,82]],[(324.0,813.0,420.0,828.0),[79,82]],[(173.0,647.0,273.0,669.0),[79,83]],[(545.0,600.0,582.0,609.0),[81,83]],[(578.0,757.0,617.0,771.0),[82,83]],[(325.0,590.0,384.0,605.0),[80,84]],[(327.0,663.0,654.0,680.0),[80,84]],[(475.0,821.0,716.0,841.0),[81,84]],[(26.0,465.0,158.0,499.0),[83,84]],[(317.0,815.0,416.0,825.0),[83,84]],[(25.0,24.0,163.0,75.0),[82,85]],[(160.0,504.0,490.0,524.0),[83,86]],[(165.0,647.0,267.0,669.0),[84,87]],[(319.0,812.0,410.0,825.0),[85,87]],[(24.0,465.0,156.0,503.0),[86,87]],[(16.0,492.0,74.0,507.0),[86,87]],[(320.0,588.0,379.0,607.0),[86,87]],[(574.0,756.0,611.0,769.0),[86,87]],[(414.0,600.0,453.0,613.0),[84,88]],[(320.0,660.0,647.0,681.0),[85,89]],[(73.0,29.0,164.0,73.0),[88,89]],[(11.0,491.0,87.0,509.0),[88,89]],[(320.0,592.0,372.0,605.0),[88,90]],[(574.0,757.0,605.0,767.0),[89,90]],[(9.0,491.0,93.0,509.0),[90,91]],[(468.0,820.0,716.0,841.0),[85,92]],[(156.0,504.0,484.0,524.0),[87,92]],[(316.0,813.0,405.0,827.0),[89,92]],[(314.0,588.0,375.0,607.0),[91,92]],[(408.0,600.0,449.0,613.0),[91,93]],[(160.0,646.0,261.0,666.0),[89,94]],[(22.0,464.0,143.0,498.0),[88,95]],[(315.0,661.0,641.0,680.0),[90,95]],[(25.0,24.0,163.0,75.0),[92,95]],[(314.0,591.0,368.0,605.0),[93,95]],[(566.0,756.0,606.0,765.0),[95,96]],[(151.0,504.0,478.0,524.0),[93,97]],[(652.0,681.0,704.0,695.0),[95,98]],[(462.0,819.0,716.0,840.0),[93,99]],[(25.0,465.0,137.0,498.0),[96,99]],[(73.0,29.0,164.0,73.0),[98,99]],[(9.0,489.0,81.0,508.0),[94,100]],[(158.0,647.0,254.0,665.0),[95,100]],[(310.0,660.0,635.0,680.0),[96,101]],[(11.0,489.0,71.0,508.0),[101,102]],[(136.0,732.0,430.0,756.0),[43,103]],[(642.0,755.0,708.0,764.0),[100,103]],[(25.0,24.0,163.0,75.0),[102,103]],[(144.0,504.0,473.0,524.0),[98,104]],[(25.0,465.0,144.0,498.0),[100,105]],[(397.0,600.0,431.0,609.0),[104,105]],[(643.0,681.0,703.0,695.0),[104,105]],[(455.0,819.0,716.0,839.0),[100,106]],[(152.0,647.0,250.0,665.0),[101,106]],[(73.0,29.0,164.0,73.0),[108,109]],[(145.0,646.0,248.0,665.0),[107,110]],[(642.0,680.0,701.0,693.0),[107,110]],[(394.0,600.0,437.0,609.0),[109,110]],[(437.0,600.0,472.0,609.0),[109,110]],[(557.0,755.0,631.0,764.0),[109,110]],[(304.0,661.0,630.0,677.0),[102,111]],[(146.0,645.0,242.0,664.0),[111,112]],[(643.0,681.0,674.0,691.0),[111,112]],[(393.0,600.0,431.0,609.0),[111,113]],[(555.0,755.0,594.0,764.0),[111,113]],[(25.0,24.0,163.0,75.0),[112,115]],[(140.0,503.0,467.0,524.0),[105,118]],[(0.0,474.0,149.0,494.0),[110,118]],[(390.0,597.0,555.0,609.0),[114,118]],[(73.0,29.0,164.0,73.0),[118,119]],[(130.0,787.0,588.0,813.0),[51,121]],[(449.0,819.0,716.0,837.0),[107,121]],[(300.0,660.0,622.0,677.0),[112,121]],[(144.0,643.0,244.0,665.0),[114,121]],[(134.0,500.0,462.0,520.0),[119,121]],[(3.0,472.0,144.0,495.0),[120,121]],[(386.0,595.0,549.0,608.0),[120,121]],[(637.0,677.0,701.0,691.0),[120,121]],[(18.0,88.0,272.0,122.0),[0,124]],[(25.0,24.0,163.0,75.0),[122,124]],[(608.0,1171.0,698.0,1216.0),[126,127]],[(570.0,1169.0,699.0,1219.0),[128,130]],[(600.0,1172.0,700.0,1217.0),[143,144]],[(28.0,462.0,177.0,503.0),[122,147]],[(570.0,1169.0,699.0,1219.0),[148,149]],[(136.0,732.0,424.0,755.0),[104,151]],[(567.0,1165.0,700.0,1221.0),[153,154]],[(32.0,467.0,184.0,500.0),[149,158]],[(570.0,1169.0,699.0,1219.0),[158,159]],[(30.0,461.0,179.0,503.0),[159,162]],[(251.0,791.0,467.0,813.0),[124,164]],[(29.0,454.0,216.0,516.0),[163,165]],[(570.0,1169.0,699.0,1219.0),[168,169]],[(30.0,462.0,185.0,504.0),[167,182]],[(30.0,462.0,179.0,503.0),[183,190]],[(224.0,784.0,495.0,812.0),[169,196]],[(599.0,1166.0,699.0,1221.0),[203,204]],[(226.0,791.0,495.0,813.0),[197,218]],[(31.0,462.0,173.0,503.0),[191,219]],[(573.0,1169.0,699.0,1219.0),[218,219]],[(28.0,464.0,175.0,497.0),[220,223]],[(602.0,1172.0,700.0,1217.0),[223,224]],[(573.0,1169.0,699.0,1219.0),[228,229]],[(600.0,1172.0,700.0,1217.0),[233,234]],[(27.0,461.0,175.0,503.0),[224,235]],[(21.0,458.0,175.0,504.0),[236,237]],[(570.0,1169.0,699.0,1219.0),[238,239]],[(570.0,1169.0,699.0,1219.0),[248,249]],[(600.0,1172.0,700.0,1217.0),[253,254]],[(599.0,1166.0,699.0,1221.0),[263,264]],[(192.0,791.0,529.0,813.0),[221,282]],[(599.0,1166.0,699.0,1221.0),[283,284]],[(573.0,1169.0,699.0,1219.0),[288,289]],[(566.0,1171.0,700.0,1219.0),[299,300]],[(602.0,1172.0,700.0,1217.0),[303,304]],[(606.0,1169.0,699.0,1219.0),[306,307]],[(570.0,1169.0,699.0,1219.0),[308,309]],[(602.0,1166.0,699.0,1221.0),[313,314]],[(599.0,1166.0,699.0,1221.0),[323,324]],[(570.0,1169.0,699.0,1219.0),[328,329]],[(582.0,1173.0,606.0,1212.0),[333,334]],[(600.0,1172.0,700.0,1217.0),[333,334]],[(218.0,788.0,504.0,815.0),[286,336]],[(570.0,1169.0,699.0,1219.0),[338,339]],[(599.0,1166.0,699.0,1221.0),[343,344]],[(570.0,1169.0,699.0,1219.0),[358,359]],[(600.0,1172.0,700.0,1217.0),[363,364]],[(573.0,1169.0,699.0,1219.0),[368,369]],[(599.0,1166.0,699.0,1221.0),[373,374]],[(578.0,1176.0,615.0,1211.0),[373,374]],[(239.0,788.0,480.0,815.0),[341,391]],[(570.0,1169.0,699.0,1219.0),[398,399]],[(564.0,1168.0,700.0,1220.0),[408,410]],[(599.0,1166.0,699.0,1221.0),[433,434]],[(564.0,1168.0,700.0,1220.0),[438,440]],[(600.0,1172.0,700.0,1217.0),[443,444]],[(205.0,789.0,515.0,812.0),[394,445]],[(573.0,1169.0,699.0,1219.0),[448,449]],[(600.0,1172.0,700.0,1217.0),[453,454]],[(26.0,460.0,175.0,503.0),[239,457]],[(570.0,1169.0,699.0,1219.0),[458,461]],[(600.0,1172.0,700.0,1217.0),[463,464]],[(570.0,1169.0,699.0,1219.0),[468,469]],[(580.0,1177.0,613.0,1209.0),[473,474]],[(600.0,1172.0,700.0,1217.0),[473,474]],[(566.0,1171.0,700.0,1219.0),[479,480]],[(602.0,1172.0,700.0,1217.0),[483,484]],[(573.0,1169.0,699.0,1219.0),[488,489]],[(580.0,1177.0,609.0,1208.0),[493,494]],[(600.0,1172.0,700.0,1217.0),[493,494]],[(251.0,787.0,467.0,813.0),[447,497]]]
    bbox_hist = [[(148.0,656.0,832.0,679.0),[0,1]],[(148.0,656.0,823.0,679.0),[2,3]],[(79.0,659.0,140.0,683.0),[0,4]],[(885.0,659.0,1267.0,676.0),[3,4]],[(153.0,656.0,816.0,679.0),[4,5]],[(153.0,656.0,805.0,679.0),[6,7]],[(153.0,659.0,788.0,678.0),[10,11]],[(148.0,656.0,773.0,679.0),[13,14]],[(148.0,656.0,767.0,679.0),[15,16]],[(148.0,656.0,756.0,679.0),[17,18]],[(816.0,658.0,1032.0,676.0),[18,19]],[(149.0,659.0,743.0,678.0),[20,21]],[(801.0,658.0,952.0,676.0),[21,22]],[(148.0,656.0,735.0,679.0),[22,24]],[(1193.0,660.0,1256.0,674.0),[23,24]],[(788.0,658.0,1005.0,676.0),[24,25]],[(153.0,656.0,723.0,679.0),[25,26]],[(777.0,658.0,989.0,676.0),[27,28]],[(148.0,656.0,709.0,679.0),[28,29]],[(148.0,656.0,697.0,679.0),[31,32]],[(759.0,658.0,980.0,676.0),[31,32]],[(148.0,656.0,688.0,679.0),[33,34]],[(1149.0,662.0,1214.0,673.0),[33,34]],[(147.0,656.0,679.0,679.0),[35,36]],[(148.0,656.0,671.0,679.0),[37,38]],[(725.0,658.0,1024.0,676.0),[39,40]],[(148.0,656.0,661.0,679.0),[39,41]],[(1195.0,656.0,1271.0,675.0),[39,41]],[(1115.0,661.0,1184.0,674.0),[40,41]],[(85.0,662.0,137.0,682.0),[5,42]],[(712.0,658.0,948.0,676.0),[41,42]],[(144.0,656.0,649.0,679.0),[42,43]],[(1108.0,661.0,1176.0,674.0),[42,43]],[(1189.0,656.0,1265.0,675.0),[42,43]],[(1097.0,662.0,1160.0,675.0),[45,46]],[(693.0,659.0,987.0,676.0),[47,48]],[(145.0,654.0,625.0,682.0),[48,49]],[(148.0,656.0,613.0,679.0),[50,52]],[(141.0,654.0,604.0,682.0),[53,54]],[(661.0,659.0,991.0,676.0),[54,55]],[(648.0,659.0,985.0,676.0),[57,58]],[(153.0,658.0,581.0,680.0),[58,59]],[(152.0,656.0,567.0,679.0),[61,62]],[(629.0,658.0,1012.0,676.0),[61,62]],[(148.0,658.0,555.0,680.0),[64,65]],[(152.0,658.0,545.0,679.0),[66,67]],[(1196.0,662.0,1268.0,675.0),[66,67]],[(151.0,658.0,532.0,679.0),[69,70]],[(597.0,659.0,980.0,678.0),[69,70]],[(79.0,662.0,141.0,682.0),[43,71]],[(1188.0,660.0,1269.0,675.0),[69,71]],[(148.0,658.0,517.0,680.0),[72,73]],[(580.0,656.0,1273.0,678.0),[72,74]],[(153.0,658.0,508.0,680.0),[74,75]],[(567.0,656.0,1271.0,679.0),[75,76]],[(153.0,658.0,500.0,680.0),[76,77]],[(153.0,658.0,491.0,680.0),[78,79]],[(551.0,654.0,1275.0,682.0),[78,79]],[(540.0,654.0,1276.0,682.0),[80,82]],[(153.0,658.0,477.0,680.0),[81,82]],[(153.0,658.0,468.0,680.0),[83,84]],[(529.0,654.0,1275.0,682.0),[83,85]],[(148.0,658.0,461.0,680.0),[85,86]],[(85.0,662.0,137.0,682.0),[72,87]],[(521.0,656.0,1273.0,679.0),[86,87]],[(148.0,658.0,443.0,680.0),[89,90]],[(512.0,659.0,905.0,678.0),[89,90]],[(983.0,655.0,1265.0,680.0),[89,90]],[(148.0,658.0,433.0,680.0),[91,92]],[(495.0,659.0,888.0,678.0),[92,93]],[(889.0,662.0,957.0,675.0),[92,93]],[(970.0,656.0,1252.0,679.0),[92,93]],[(153.0,658.0,421.0,680.0),[94,95]],[(484.0,654.0,1244.0,682.0),[94,95]],[(159.0,658.0,415.0,680.0),[96,97]],[(153.0,658.0,405.0,680.0),[98,99]],[(947.0,656.0,1225.0,679.0),[98,99]],[(153.0,658.0,393.0,680.0),[101,102]],[(452.0,656.0,1209.0,679.0),[102,103]],[(153.0,658.0,383.0,680.0),[103,104]],[(148.0,658.0,375.0,680.0),[105,106]],[(435.0,654.0,1196.0,682.0),[105,106]],[(79.0,659.0,140.0,683.0),[90,109]],[(147.0,658.0,361.0,680.0),[108,109]],[(424.0,656.0,1184.0,680.0),[108,109]],[(1248.0,659.0,1273.0,675.0),[108,109]],[(148.0,658.0,353.0,680.0),[110,111]],[(412.0,654.0,1175.0,682.0),[110,111]],[(73.0,659.0,152.0,683.0),[111,112]],[(1231.0,656.0,1277.0,680.0),[111,112]],[(405.0,654.0,1168.0,682.0),[112,113]],[(71.0,657.0,341.0,682.0),[113,114]],[(1223.0,656.0,1277.0,680.0),[113,114]],[(57.0,659.0,77.0,676.0),[113,116]],[(71.0,657.0,332.0,680.0),[115,116]],[(391.0,654.0,1153.0,682.0),[115,116]],[(1216.0,658.0,1276.0,678.0),[115,116]],[(383.0,654.0,1145.0,682.0),[117,118]],[(1205.0,655.0,1275.0,681.0),[117,118]],[(149.0,658.0,319.0,680.0),[118,119]],[(1196.0,656.0,1272.0,679.0),[119,120]],[(371.0,656.0,1131.0,680.0),[120,121]],[(153.0,658.0,309.0,680.0),[120,121]],[(73.0,659.0,151.0,683.0),[118,122]],[(1188.0,656.0,1275.0,679.0),[121,122]],[(361.0,654.0,1124.0,682.0),[122,123]],[(72.0,656.0,299.0,684.0),[123,124]],[(353.0,654.0,1116.0,682.0),[124,125]],[(1173.0,656.0,1271.0,679.0),[124,125]],[(147.0,658.0,288.0,680.0),[125,126]],[(1065.0,56.0,1207.0,82.0),[72,127]],[(73.0,656.0,151.0,686.0),[125,127]],[(337.0,654.0,1101.0,682.0),[127,128]],[(164.0,658.0,276.0,680.0),[128,129]],[(1156.0,658.0,1275.0,677.0),[128,129]],[(1146.0,657.0,1276.0,678.0),[130,131]],[(73.0,659.0,141.0,683.0),[128,132]],[(148.0,658.0,257.0,680.0),[132,133]],[(317.0,656.0,1079.0,680.0),[132,133]],[(1138.0,658.0,1276.0,677.0),[132,133]],[(1129.0,657.0,1276.0,678.0),[134,135]],[(1121.0,657.0,1275.0,678.0),[136,137]],[(148.0,658.0,232.0,680.0),[138,139]],[(1112.0,657.0,1276.0,678.0),[138,139]],[(149.0,658.0,223.0,680.0),[140,141]],[(1104.0,657.0,1273.0,678.0),[140,141]],[(73.0,659.0,143.0,683.0),[140,143]],[(276.0,658.0,1033.0,680.0),[142,143]],[(155.0,659.0,213.0,679.0),[142,143]],[(1095.0,658.0,1273.0,676.0),[142,143]],[(1085.0,657.0,1271.0,678.0),[144,145]],[(264.0,658.0,1020.0,680.0),[145,146]],[(73.0,659.0,152.0,683.0),[144,147]],[(255.0,658.0,1012.0,680.0),[147,148]],[(70.0,658.0,190.0,683.0),[148,149]],[(57.0,659.0,77.0,676.0),[148,149]],[(1069.0,657.0,1276.0,678.0),[148,149]],[(1064.0,50.0,1206.0,85.0),[128,150]],[(73.0,659.0,155.0,683.0),[150,151]],[(1061.0,657.0,1275.0,678.0),[150,151]],[(1053.0,656.0,1275.0,679.0),[152,153]],[(223.0,658.0,980.0,680.0),[154,155]],[(1044.0,656.0,1275.0,679.0),[154,155]],[(1036.0,656.0,1276.0,679.0),[156,157]],[(212.0,658.0,968.0,680.0),[157,158]],[(1028.0,656.0,1273.0,679.0),[158,159]],[(201.0,658.0,959.0,680.0),[159,160]],[(1019.0,656.0,1269.0,679.0),[160,161]],[(57.0,658.0,80.0,676.0),[155,163]],[(185.0,658.0,944.0,680.0),[163,164]],[(1005.0,656.0,1276.0,679.0),[163,164]],[(177.0,658.0,933.0,680.0),[165,166]],[(997.0,656.0,1273.0,679.0),[165,166]],[(57.0,658.0,80.0,676.0),[165,168]],[(988.0,656.0,1268.0,679.0),[167,168]],[(980.0,656.0,1260.0,679.0),[169,170]],[(73.0,659.0,143.0,683.0),[152,171]],[(152.0,656.0,913.0,680.0),[170,171]],[(151.0,656.0,904.0,680.0),[172,173]],[(965.0,657.0,1275.0,678.0),[172,173]],[(147.0,656.0,896.0,680.0),[174,175]],[(956.0,656.0,1275.0,679.0),[174,175]],[(149.0,656.0,887.0,680.0),[176,177]],[(948.0,656.0,1275.0,679.0),[176,177]],[(147.0,656.0,880.0,680.0),[178,179]],[(937.0,656.0,1275.0,679.0),[178,180]],[(149.0,659.0,868.0,678.0),[180,181]],[(927.0,659.0,1273.0,678.0),[181,182]],[(147.0,656.0,861.0,680.0),[182,183]],[(917.0,656.0,1273.0,679.0),[183,184]],[(147.0,656.0,852.0,680.0),[184,185]],[(911.0,656.0,1272.0,679.0),[185,186]],[(144.0,656.0,843.0,680.0),[186,187]],[(79.0,659.0,141.0,683.0),[172,188]],[(901.0,659.0,1267.0,678.0),[187,188]],[(147.0,656.0,836.0,680.0),[188,189]],[(78.0,656.0,150.0,685.0),[189,190]],[(893.0,659.0,1257.0,678.0),[189,190]],[(71.0,656.0,820.0,682.0),[192,193]],[(879.0,659.0,1272.0,678.0),[192,193]],[(868.0,656.0,1271.0,678.0),[194,195]],[(147.0,659.0,803.0,678.0),[195,196]],[(860.0,656.0,1271.0,678.0),[196,197]],[(73.0,659.0,152.0,683.0),[195,198]],[(149.0,659.0,796.0,678.0),[197,198]],[(851.0,656.0,1273.0,678.0),[198,199]],[(841.0,656.0,1273.0,678.0),[200,201]],[(77.0,659.0,148.0,683.0),[202,203]],[(832.0,656.0,1275.0,678.0),[202,204]],[(77.0,659.0,141.0,683.0),[204,206]],[(149.0,659.0,761.0,678.0),[205,206]],[(820.0,656.0,1275.0,678.0),[205,206]],[(149.0,659.0,751.0,678.0),[207,208]],[(811.0,656.0,1275.0,678.0),[207,208]],[(77.0,656.0,140.0,685.0),[208,210]],[(148.0,658.0,743.0,680.0),[209,210]],[(803.0,656.0,1275.0,678.0),[209,210]],[(789.0,656.0,1275.0,678.0),[212,214]],[(77.0,656.0,145.0,685.0),[213,214]],[(148.0,658.0,725.0,680.0),[213,214]],[(776.0,656.0,1273.0,678.0),[215,216]],[(147.0,658.0,713.0,680.0),[216,217]],[(77.0,659.0,153.0,683.0),[216,217]],[(767.0,656.0,1273.0,678.0),[217,219]],[(77.0,659.0,147.0,683.0),[218,219]],[(155.0,659.0,699.0,678.0),[219,220]],[(755.0,656.0,1273.0,678.0),[220,221]],[(152.0,658.0,692.0,680.0),[221,222]],[(747.0,659.0,1269.0,678.0),[222,223]],[(152.0,659.0,681.0,678.0),[223,224]],[(740.0,659.0,1265.0,678.0),[224,225]],[(147.0,656.0,673.0,680.0),[225,226]],[(77.0,659.0,141.0,683.0),[220,228]],[(721.0,659.0,1267.0,678.0),[228,229]],[(149.0,659.0,655.0,678.0),[229,230]],[(79.0,659.0,143.0,683.0),[230,231]],[(147.0,656.0,648.0,680.0),[231,232]],[(59.0,659.0,77.0,676.0),[233,234]],[(149.0,656.0,640.0,680.0),[233,234]],[(147.0,656.0,631.0,680.0),[235,236]],[(152.0,656.0,617.0,680.0),[238,239]],[(672.0,656.0,1205.0,678.0),[239,240]],[(148.0,574.0,403.0,601.0),[107,241]],[(147.0,656.0,608.0,680.0),[240,241]],[(149.0,617.0,948.0,639.0),[107,242]],[(73.0,659.0,143.0,683.0),[233,242]],[(147.0,656.0,601.0,680.0),[242,243]],[(656.0,656.0,1271.0,679.0),[243,244]],[(147.0,654.0,587.0,682.0),[245,246]],[(79.0,659.0,141.0,683.0),[245,246]],[(641.0,656.0,1275.0,679.0),[246,248]]]
    feature_frame_idx = [72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249]

    frame_to_subtitle = {k:[] for k in range(len(frames[750:1000]))}
    frame_to_name = {k:[] for k in range(len(frames[750:1000]))}

    for bbox in bbox_hist:
        frame_range = bbox[-1]
        num_frames = frame_range[-1] - frame_range[0]
        # Detect subtitle bbox
        frame_to_subtitle = find_subtitle_bbox(frame_to_subtitle, frame_range, num_frames, bbox, fps, height)

        # Detect name bbox
        frame_to_name = find_name_bbox(frame_to_name, frame_range, num_frames, bbox, fps, height, feature_frame_idx)

    # Remove any duplicates
    frame_to_name = remove_duplicate_bbox(frame_to_subtitle, frame_to_name)

    # Plot bboxes and output video
    for idx, frame in enumerate(frames[750:1000]):
        img = cv2.imread(frame)

        # Plot subtitle bboxes
        for bbox in frame_to_subtitle[idx]:
            img = plot_bbox(img, 'Subtitle', bbox, (0, 255, 0))
        
        # Plot name bboxes (highest found)
        if len(frame_to_name[idx]) >= 1:
            highest_y = height
            for bbox in frame_to_name[idx]:
                if bbox[-1] < highest_y:
                    highest_y = bbox[-1]
                    highest_bbox = bbox
            img = plot_bbox(img, 'Name', highest_bbox, (255, 0, 0))
        out.write(img)

    out.release()
    cap.release()

    print('BBOX', bbox_hist)
    print('FEATURE FRAME IDX', feature_frame_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--frame_save_path", type=str)
    parser.add_argument("--lang", type=str, default='eng', choices=['eng', 'ch'])

    args = parser.parse_args()
    main(args)

# for bbox in subtitle_bboxes:
#     frame_range = bbox[-1]
#     for frame_num in frame_range:
#         img = cv2.imread('{}/{}_{}.jpg'.format(frame_save_path, base_filename, frame_num))
#         cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][-1]), (0, 255, 0))

    # r = sr.Recognizer()
    # temp = sr.AudioFile('{}/temp.wav'.format(asr_temp_path))
    # with temp as source:
    #     temp_audio = r.record(source)

    # if lang == 'eng':
    #     try:
    #         pred_text = r.recognize_google(temp_audio)#, language='en-SG')
    #     except sr.UnknownValueError:
    #         pred_text = 'UNKNOWN'
    # elif lang == 'ch':
    #     try:
    #         pred_text = r.recognize_google(temp_audio, language='zh')
    #     except sr.UnknownValueError:
    #         pred_text = 'UNKNOWN'
   

