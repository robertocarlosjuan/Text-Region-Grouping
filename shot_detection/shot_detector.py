import cv2
import os, sys
import numpy as np
import torch

import shot_detection.gooleNet_KTS as google_kts
from pathlib import Path

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def sample_img_from_shot(cps,videoFile,out_prefix, idx):
    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cps[0])

    img_idx=int((cps[0]+cps[1])/2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, img_idx)
    ret, frame = cap.read()
    rlt_img_file=out_prefix+f'_{idx}'+".jpg"
    cv2.imwrite(rlt_img_file, frame)
    
    
def shot2Video(cps,videoFile,shot_file):
    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cps[0])

     # create summary video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(shot_file, fourcc, fps, (width, height))
    for i in range(cps[1]-cps[0]+1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()
    

def shot_to_video_audio(cps, video_file, shot_file):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_time = cps[0] / fps
    end_time = cps[1] / fps
    print('CPS', cps)
    print('TIME', start_time, end_time)
    ffmpeg_extract_subclip(video_file, start_time, end_time, targetname=shot_file)

    
    
def shot2Frames(cps,videoFile,out_dir):
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    cap = cv2.VideoCapture(videoFile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cps[0])
    stem=Path(videoFile).stem
    for i in range(cps[1]-cps[0]+1):
        ret, frame = cap.read()
        if not ret:
            break
        idx=i+cps[0]
        filename=os.path.join(out_dir,f'{stem}_{idx}.jpg')
        cv2.imwrite(filename,frame)
    cap.release()


def shot_key_frame_detect(input_path, outPath):
    if os.path.isdir(input_path):
        video_names=os.listdir(input_path)
        video_names=[vname for vname in video_names if vname.endswith('.mp4')]
    else:
        video_names=[os.path.basename(input_path)]
        input_path= os.path.dirname(input_path)
    
    sample_rate = 2
    Path(outPath).mkdir(exist_ok=True, parents=True)
    video_proc = google_kts.VideoPreprocessor(sample_rate)
    for video in video_names:
        videoname=f'{input_path}/{video}'
        out_prefix=outPath+'/'+Path(video).stem
        rlt_seg_file = out_prefix +".txt"
        if os.path.exists(rlt_seg_file):       #already exist
            continue
        print(f'short detection from {video}...')
        n_frames, cps, nfps,picks = video_proc.seg_shot_detect(videoname, batch_size=100, max_len=2000, max_overlap=100, vmax=1.5*sample_rate)
        seg_num = len(cps)
  #save shots' locations
        with open(rlt_seg_file,'wt') as f:
            for n in range(seg_num): 
                start=cps[n][0] 
                end=cps[n][1]
                line=f'{n} {start} {end }\n'
                f.write(line)
        #save key frames of each shot
        for i in range(seg_num):
            sample_img_from_shot(cps[i], videoname, out_prefix, i)

def video_image_main(video_file):
     _, videoname=os.path.split(video_file)
#  videoname,_=os.path.splittext(videoname)     #alternative appraoch
#   videoname = videoname.stem
     outPath = './shotRlt'
     videoname=videoname.split('.')[0]
     out_prefix=outPath+'/'+videoname
     rlt_seg_file = out_prefix +".txt"
     if os.path.exists(rlt_seg_file):       #already exist
        return

     sample_rate = 2
     Path(outPath).mkdir(exist_ok=True, parents=True)
     video_proc = google_kts.VideoPreprocessor(sample_rate)
 
# shot detection 
     n_frames, cps, nfps,picks = video_proc.seg_shot_detect(video_file, batch_size=100, max_len=2000, max_overlap=100, vmax=1.5*sample_rate)
     seg_num = len(cps)
#save shots' locations
     with open(rlt_seg_file,'wt') as f:
         for n in range(seg_num): 
             start=cps[n][0] 
             end=cps[n][1]
             line=f'{n} {start} {end }\n'
             f.write(line)

     if not os.path.exists(outPath):
         os.makedirs(outPath)
     os.makedirs('{}/video'.format(outPath), exist_ok=True)
     os.makedirs('{}/image'.format(outPath), exist_ok=True)
     for i in range(seg_num):
         #save video shots
         video_outFile='{}/video/{}_{}.mp4'.format(outPath,videoname,i)
         shot_to_video_audio(cps[i],video_file, video_outFile)
        #  shot2Video(cps[i],video_file,outFile)
         #save key frames of each shot
         image_out_prefix = '{}/image/{}'.format(outPath, videoname)
         sample_img_from_shot(cps[i], video_file, image_out_prefix, i)


def video_main(video_file, output_mode='video_seg'):
     _, videoname=os.path.split(video_file)
#  videoname,_=os.path.splittext(videoname)     #alternative appraoch
#   videoname = videoname.stem
     videoname=videoname.split('.')[0]
     out_prefix=outPath+'/'+videoname
     rlt_seg_file = out_prefix +".txt"

# shot detection 
     n_frames, cps, nfps,picks = video_proc.seg_shot_detect(video_file, batch_size=100, max_len=2000, max_overlap=100, vmax=1.5*sample_rate)
     seg_num = len(cps)
#save shots' locations
     with open(rlt_seg_file,'wt') as f:
           for n in range(seg_num): 
               start=cps[n][0]
               end=cps[n][1]
               line=f'{n} {start} {end }\n'
               f.write(line)
# #save video shots
     if not os.path.exists(outPath):
          os.makedirs(outPath)
     if output_mode == 'video_seg':      #save segment into video chips
          for i in range(seg_num):
              outFile=out_prefix+f'_{i}'+".mp4"
              shot2Video(cps[i],video_file,outFile)
     elif output_mode == 'video_frame':    #save frames of segments into respective folders
         for i in range(seg_num):
              subfolder = os.path.join(outPath, f'{videoname}-{i}')
              Path(subfolder).mkdir(parents=True, exist_ok=True)
              shot2Frames(cps[i],video_file,subfolder)
            


def img_main(video_file):
#check video length, temporarily ignore videos longer than 23,000 frames
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    #     _, videoname=os.path.split(video_file)
    #     videoname=videoname.split('.')[0]
    videoname = Path(video_file).stem
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    out_prefix = outPath + "/"+videoname    
    rlt_seg_file = out_prefix+".txt"
        
# shot detection
    n_frames, cps, nfps,picks = video_proc.seg_shot_detect(video_file, batch_size=100, max_len=2000,max_overlap=100, vmax=1.5*sample_rate)
    seg_num = len(cps)

#save shots' locations
    with open(rlt_seg_file,'wt') as f:
        for n in range(seg_num): 
            start=cps[n][0] 
            end=cps[n][1]
            line=f'{n} {start} {end }\n'
            f.write(line)
#save key frames of each shot
    for i in range(seg_num):
        sample_img_from_shot(cps[i], video_file, out_prefix)
         
if __name__ == '__main__':
    mode = sys.argv[1]
    video_path = sys.argv[2]
    outPath = sys.argv[3]
    sample_rate = int(sys.argv[4])
    Path(outPath).mkdir(exist_ok=True, parents=True)
    video_proc = google_kts.VideoPreprocessor(sample_rate)
    
    video_names=os.listdir(video_path)
    video_names=[vname for vname in video_names if vname.endswith('.mp4')]
    for video in video_names:
        print(f'Preprocessing video {video}...\n')
        if mode == 'video':
           video_main(video_path+'/'+video,'video_seg')
           
        if mode == "image":
            img_main(video_path+'/'+video)
