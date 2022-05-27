import argparse
from classes import run

# video_path = "/home/hcari/trg/videos/"
# audio_save_path = "/home/hcari/trg/visualize/wav_files"
# video_path = '../montage_detection/sinovac_ver/ref_video/nov_3jnLn2w.mp4'
# audio_save_path = '../non_commit_trg/audio'
# shot_path = '../montage_detection/images/sinovac/db'
# out_dir = '../non_commit_trg/output'

# python classes.py --video_path /home/hcari/trg/videos/ --audio_save_path /home/hcari/trg/visualize/wav_files --shot_path /home/hcari/trg/visualize/shot/ --out_dir /home/hcari/trg/visualize

def parse_args():
    parser = argparse.ArgumentParser(description='Text Region Grouping')
    parser.add_argument('--video_path', type=str, help='path to folder of videos')
    parser.add_argument('--audio_save_path', type=str, help='path to folder to save audio files')
    parser.add_argument('--shot_path', type=str, help='path to folder to save shots')
    parser.add_argument('--out_dir', type=str, help='path to folder to save videos')
    parser.add_argument('--xml_dir', type=str, help='path to folder to save xml file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run(args.video_path, args.audio_save_path, args.shot_path, args.out_dir, args.xml_dir)