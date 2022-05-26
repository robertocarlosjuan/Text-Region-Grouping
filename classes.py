import os
import cv2
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_silence

from config import *
from utils import *
from sklearn.cluster import KMeans
from shapely.geometry import Polygon
from Levenshtein import distance as lev
from PIL import Image
import sys
# sys.path.append("/home/hcari/trg/image_quality/")
# import imquality.brisque as brisque

from shot_detection.shot_detector import shot_key_frame_detect

class BoundingBoxInstance:
    def __init__(self, coords, frame, text, conf, type=None):
        self.coords = coords # [[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]]
        self.frame = frame
        self.text = text
        self.conf = conf
        self.type = type
        self.bbox = None
        self.cropped_image = None
        self.colour = None
        self.image_quality_score = None

    def is_horizontal(self):
        return abs(self.coords[0][1] - self.coords[1][1]) <= hor_leeway and abs(self.coords[2][1] - self.coords[3][1]) <= hor_leeway

    def overlaps(self, new_coords):
        overlap(self.coords, new_coords)

    def get_cropped_image(self, frame, coords):
        plot_coords = get_plot_coords(coords)
        self.cropped_image = frame.image[int(plot_coords[1]):int(plot_coords[3]), int(plot_coords[0]):int(plot_coords[2]), :]
        return self.cropped_image

    def get_image_quality_score(self, coords):
        cropped_image = self.get_cropped_image(self.frame, coords)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_image)
        try:
            self.image_quality_score = brisque.score(pil_image)
            return self.image_quality_score
        except:
            return None

    def get_common_colours(self): # K-Means, K=2, check sse
        if self.cropped_image is None:
            coords = self.bbox.overall_coords if self.bbox is not None else self.coords
            self.get_cropped_image(self.frame, coords)
        points = self.cropped_image.copy().reshape(-1, 3) / 255
        clt = KMeans(n_clusters=2)
        clt.fit(points)
        clusters = []
        labels = clt.labels_
        for label_type in [0,1]:
            points_of_cluster = points[labels==label_type,:]
            centroid_of_cluster = np.mean(points_of_cluster, axis=0)
            clusters.append(centroid_of_cluster * 255)
        self.colour = clusters
        return clusters

class BoundingBox:
    def __init__(self, video, type=None):
        self.video = video
        self.bboxes = []
        self.type = type
        self.overall_coords = None
        self.overall_colours = None
        self.colour_cluster = {0:[], 1:[]}
        self.prop_zero = None
        self.density = None
        self.texts = None
        self.text_length = None
        self.consider_merging = None
        self.image_quality_score = None

    def get_frame_nos(self):
        return [bbox_inst.frame.frame_no for bbox_inst in self.bboxes]

    def match(self, bbox_instance):
        if self.overlaps(bbox_instance) and self.temporally_near(bbox_instance):
            return True

    def add_bbox_instance(self, bbox_instance, frame):
        self.bboxes.append(bbox_instance)
        # self.update_colours(bbox_instance)
        bbox_instance.bbox = self
        self.overall_coords = self.get_overall_coords()

    def check_n_add_bbox_instance(self, bbox_instance):
        if self.match(bbox_instance):
            # print(bbox_instance.text, self.bboxes[0].text)
            self.add_bbox_instance(bbox_instance, bbox_instance.frame)
            return True
        else:
            return False

    def overlaps(self, bbox_instance):
        return overlap(self.get_overall_coords(), bbox_instance.coords)

    def merge_bounding_box(self, new_bbox):
        if overlap(self.get_overall_coords(), new_bbox.get_overall_coords()):
            self.bboxes.extend(new_bbox.bboxes)
            for bbox_inst in new_bbox.bboxes:
                bbox_inst.bbox = self
            self.overall_coords = self.get_overall_coords()
            return True
        else:
            return False

    def temporally_near(self, bbox_instance):
        new_bbox_inst_frame_no = bbox_instance.frame.frame_no
        frame_dists = [abs(new_bbox_inst_frame_no-bbox_inst.frame.frame_no) for bbox_inst in self.bboxes]
        return min(frame_dists) < temporal_dist_thres

    def get_overall_coords(self):
        overall_coords = self.bboxes[0].coords # Union of all bbox instances
        for bbox in self.bboxes:
            overall_coords  = get_union_region(overall_coords, bbox.coords)
        self.overall_coords = overall_coords
        return overall_coords

    def check_channel(self):
        if self.type == "channel":
            return True
        elif self.type is None and not self.changing_texts() and self.check_bbox_at_corner() and (self.check_bbox_last_through_vid() or self.is_a_known_channel_name()):
            self.type = "channel"
            return True
        else:
            return False
    
    def check_title(self):
        if self.type == "title":
            return True
        elif self.type is None and not self.changing_texts() and self.check_bbox_last_through_vid():
            self.type = "title"
        else:
            return False

    def check_static_type(self, not_channel=False):
        # print()
        # print(self.bboxes[0].text, self.overall_coords)
        known_channel = self.is_a_known_channel_name()
        if not self.is_horizontal():
            # print("Not horizontal:  scene text")
            self.type = "scene text"
            self.consider_merging = False
        elif known_channel and self.check_bbox_at_corner():
            self.video.channel_name = known_channel
            # print("known channel: channel")
            self.type = "channel"
            self.consider_merging = False
        else:
            if self.check_bbox_lower() and self.long_texts():
                # print("lower+long: consider merging")
                self.consider_merging = True
            if not self.changing_texts():
                # print("static text")
                # print("bbox at corner: ", self.check_bbox_at_corner())
                # print("last through vid: ", self.check_bbox_last_through_vid())
                # print(self.check_bbox_at_corner(), self.check_bbox_last_through_vid())
                if known_channel is None and not not_channel and self.check_bbox_at_corner() and self.check_bbox_last_through_vid():
                    # print("channel")
                    self.video.possible_channel.append(self)
                elif not self.check_bbox_lower(): # and self.get_image_quality_score() is not None and self.image_quality_score < max_quality_scene_text:
                    # print("scene text")
                    self.type = "scene text"
                elif self.check_bbox_lower() and self.long_texts():
                    # print("bbox lower: topic sentence")
                    self.type = "topic sentence"
            elif not self.long_texts():
                self.type = "scene text"
        # elif ASR HERE
        return self.type is not None

    def long_texts(self):
        self.texts = [bbox_inst.text for bbox_inst in self.bboxes]
        self.avg_text_length = sum([len(t) for t in self.texts])/len(self.texts)
        if self.avg_text_length > min_avg_length_texts_topicrc:
            return self.texts
        else:
            return False

    def get_image_quality_score(self):
        for bbox in self.bboxes:
            img_quality = bbox.get_image_quality_score(self.overall_coords)
            if img_quality is not None:
                self.image_quality_score = img_quality
                break
        return self.image_quality_score

    def is_a_known_channel_name(self):
        self.video.known_channels = load_known_channels(known_channels_path)
        for bbox_inst in self.bboxes:
            for known_channel in self.video.known_channels:
                edit_similarity = norm_edit_sim(bbox_inst.text.lower().strip(), known_channel)
                if edit_similarity > channel_edit_sim:
                    return known_channel
        return None
        
    def check_bbox_last_through_vid(self):
        no_frame_in_bbox = len(set([bbox_inst.frame.frame_no for bbox_inst in self.bboxes]))
        no_frame_in_video = len(self.video.frames)
        prop_bbox_video_temporally = no_frame_in_bbox/no_frame_in_video
        if prop_bbox_video_temporally>=prop_bbox_video_temporally_thres:
            return True
        else:
            return False

    def check_bbox_at_corner(self):
        return check_bbox_at_edge(self.overall_coords, self.video.height, self.video.width, corner_threshold)

    def check_bbox_lower(self):
        if all(edge[1] > (1-lower_height_threshold)*self.video.height for edge in self.overall_coords):
            return True
        else:
            return False

    # def consider_merging(self):
    #     criterias = []
    #     assert self.static_consider_merging is not None
    #     criterias.append(self.static_consider_merging)
    #     return all(criterias)

    def changing_texts(self):
        texts = [bbox_inst.text for bbox_inst in self.bboxes]
        self.prop_zero = diff_density(texts)
        # print("prop_zero: ", self.prop_zero)
        return self.prop_zero >= prop_zero_thres

    def is_horizontal(self):
        return abs(self.overall_coords[0][1] - self.overall_coords[1][1]) <= hor_leeway and abs(self.overall_coords[2][1] - self.overall_coords[3][1]) <= hor_leeway

    def assign_colour_cluster(self, instance_colours):
        distances = [] # 0,0 0,1 1,0 1,1
        for i in [0,1]:
            for j, col in enumerate(instance_colours):
                distances.append((get_distance(self.overall_colours[i], col),(i, j)))
        overall_idx, instance_idx = min(distances)[1]
        return overall_idx, instance_idx

    def update_colours(self, bbox_instance):
        colours = bbox_instance.get_common_colours()
        if self.overall_colours is None:
            self.overall_colours = colours
            if get_distance((255,255,255), colours[0]) < get_distance((255,255,255), colours[1]):
                white = colours[0]
                non_white = colours[1]
            else:
                white = colours[1]
                non_white = colours[0]
            self.colour_cluster[0].append(white)
            self.colour_cluster[1].append(non_white)
        else:
            # assign to colour cluster
            i, j = self.assign_colour_cluster(colours)
            self.colour_cluster[i].append(colours[j])
            self.colour_cluster[1-i].append(colours[1-j])
            # update centroids
            self.overall_colours[i] = get_centroid(self.colour_cluster[i])
            self.overall_colours[1-i] = get_centroid(self.colour_cluster[1-i])
        return self.overall_colours

    def colour_density(self):
        points_within_epsilon = {0:0, 1:0}
        for point in self.colour_cluster[0]:
            if get_distance(self.overall_colours[0], point) <= colour_density_epsilon:
                points_within_epsilon[0] += 1
        for point in self.colour_cluster[1]:
            if get_distance(self.overall_colours[1], point) <= colour_density_epsilon:
                points_within_epsilon[1] += 1
        density = [points_within_epsilon[x]/len(self.colour_cluster[x]) for x in [0,1]]
        self.density = density
        if density[0] > min_colour_density and density[1] > min_colour_density:
            return True
        else:
            False

    def colour_match(self, bbox_instance):
        instance_colour = bbox_instance.colour if bbox_instance.colour is not None else bbox_instance.get_common_colours()
        overall_idx, instance_idx = self.assign_colour_cluster(instance_colour)
        dist = []
        dist.append(get_distance(self.overall_colours[overall_idx], instance_colour[instance_idx]))
        dist.append(get_distance(self.overall_colours[1-overall_idx], instance_colour[1-instance_idx]))
        if all(x<color_dist_thres for x in dist):
            return True
        else:
            return False
    

class Frame: # add time
    def __init__(self, image, frame_no, height=None, width=None, channels=None):
        self.image = image
        self.image_w_bbox = image
        self.frame_no = frame_no
        if height is None or width is None or channels is None:
            self.height, self.width, self.channels = image.shape
        else:
            self.height = height
            self.width = width
            self.channels = channels
        self.bbox_instances = self.generate_bbox_instances()
        self.bboxes = []

    def generate_bbox_instances(self):
        # ocr_bboxes will be a list, each item contains bounding box, text and recognition confidence
        # E.g.[[[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]],...]
        ocr_bboxes = ocr.ocr(self.image, cls=True)
        bboxes = []
        for ocr_bbox in ocr_bboxes:
            coords = ocr_bbox[0]
            text = ocr_bbox[1][0]
            conf = ocr_bbox[1][1]
            bboxes.append(BoundingBoxInstance(coords, self, text, conf))
        return bboxes
    
    def generate_bbox_list(self):
        self.bboxes = []
        for bbox_instance in self.bbox_instances:
            if bbox_instance.bbox is not None and bbox_instance.bbox not in self.bboxes:
                self.bboxes.append(bbox_instance.bbox)

class BoundingBoxGroup:
    def __init__(self, bbox, video):
        self.video = video
        self.coords = bbox.overall_coords
        if bbox.texts is None:
            bbox.long_texts()
        self.texts = bbox.texts
        self.bboxes = [bbox]
        self.overall_colours = None
        self.colour_cluster = {0:[], 1:[]}
    
    def vert_overlaps(self, bbox):
        return get_vertical_iou(bbox.overall_coords, self.coords) > vertical_iou_thres

    def add_bbox(self, bbox):
        self.bboxes.append(bbox)
        if bbox.texts is None:
            bbox.long_texts()
        self.texts.extend(bbox.texts)
        self.coords = get_horizontal_union_region(self.coords, bbox.overall_coords)
        # self.update_colours(bbox)

    def checknadd_bbox(self, bbox):
        if bbox.consider_merging is True and self.vert_overlaps(bbox): # and self.colour_match(bbox):
            self.add_bbox(bbox)
            # self.update_colours(bbox)
            return True
        return False

    def merge_bboxg(self, new_bboxg):
        if get_vertical_iou(self.coords, new_bboxg.coords) > vertical_iou_thres:
            self.bboxes.extend(new_bboxg.bboxes)
            self.texts.extend(new_bboxg.texts)
            self.coords = get_horizontal_union_region(self.coords, new_bboxg.coords)
            return True
        else:
            return False

    def check_rc(self):
        checks = []
        # print("In RC: ", self.texts[0], self.coords)
        checks.append(self.changing_texts())
        checks.append(self.check_rc_by_text() or self.check_rc_by_length())
        # print("rc checks: ", checks)
        return all(checks)

    def check_rc_by_text(self):
        rc_sorted = sorted(self.bboxes, key=lambda x: len(x.texts[0]))
        clusters = {}
        for item in rc_sorted:
            item_in_clusters = False
            for key in clusters.keys():
                if check_in(item.texts[0], key.texts[0]):
                    clusters[key] += 1
                    item_in_clusters = True
                    break
            if not item_in_clusters:
                for i in range(len(rc_sorted)-1,0,-1):
                    key = rc_sorted[i]
                    if check_in(item.texts[0], key.texts[0]) and key not in clusters.keys():
                        clusters[key] = 1
                        item_in_clusters = True
                        break
        # print(clusters)
        refined_clusters = [(k,v) for k, v in clusters.items()] # if v>1 and v!=max(clusters.values())
        return len(refined_clusters)>0

    def check_rc_by_length(self):
        # print("length of rc: ", len("".join(set(self.texts)))/len(self.video.frames))
        return len("".join(set(self.texts)))/len(self.video.frames) > min_length_texts_for_rc

    def changing_texts(self):
        self.prop_zero = diff_density(self.texts)
        # print("prop_zero: ", self.prop_zero)
        return self.prop_zero >= prop_zero_group_thres

    def checknupdate_rc(self):
        if self.check_rc():
            for bbox in self.bboxes:
                bbox.type = "rolling news"
                bbox.overall_coords = self.coords

    def assign_colour_cluster(self, bbox_colours):
        distances = [] # 0,0 0,1 1,0 1,1
        for i in [0,1]:
            for j, col in enumerate(bbox_colours):
                distances.append((get_distance(self.overall_colours[i], col),(i, j)))
        overall_idx, bbox_idx = min(distances)[1]
        return overall_idx, bbox_idx

    def update_colours(self, bbox):
        colours = bbox.overall_colours
        if self.overall_colours is None:
            self.overall_colours = colours
            if get_distance((255,255,255), colours[0]) < get_distance((255,255,255), colours[1]):
                white = colours[0]
                non_white = colours[1]
            else:
                white = colours[1]
                non_white = colours[0]
            self.colour_cluster[0].append(white)
            self.colour_cluster[1].append(non_white)
        else:
            # assign to colour cluster
            i, j = self.assign_colour_cluster(colours)
            self.colour_cluster[i].append(colours[j])
            self.colour_cluster[1-i].append(colours[1-j])
            # update centroids
            self.overall_colours[i] = get_centroid(self.colour_cluster[i])
            self.overall_colours[1-i] = get_centroid(self.colour_cluster[1-i])
        return self.overall_colours

    def colour_match(self, bbox):
        if self.overall_colours is None:
            return True
        colour = bbox.overall_colours
        overall_idx, bbox_idx = self.assign_colour_cluster(colour)
        dist = []
        dist.append(get_distance(self.overall_colours[overall_idx], colour[bbox_idx]))
        dist.append(get_distance(self.overall_colours[1-overall_idx], colour[1-bbox_idx]))
        # print("Colour dist:", bbox.bboxes[0].text, self.bboxes[0].bboxes[0].text, dist)
        if all(x<color_dist_thres for x in dist):
            return True
        else:
            return False

class Video:
    def __init__(self, video_path, audio_save_path, shot_path):
        self.video_path = video_path
        self.audio_save_path = audio_save_path
        self.base_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        self.bboxes = None
        self.height = None
        self.width = None
        self.frames = self.generate_frames()
        self.lang = 'ch'
        self.bbox_groups = None
        self.channel_name = None
        self.possible_channel = []
        self.known_channels = None
        self.shot_path = shot_path
        self.org_fps = None

    def generate_frames(self):
        vidcap = cv2.VideoCapture(self.video_path)
        success=True
        frame_no = 0
        channels = None
        frames = []
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_no*sample_rate)) 
            success,image = vidcap.read()
            if success:
                if (self.height is None) or (self.width is None):
                    self.height, self.width, channels = image.shape
                frame = Frame(image, frame_no, self.height, self.width, channels)
                frames.append(frame)
                frame_no += 1
        assert (self.height is not None) and (self.width is not None)
        vidcap.release()
        return frames

    def generate_bboxes(self):
        bboxes = []
        for frame in self.frames:
            for bbox_instance in frame.bbox_instances:
                added = False
                for recorded_bbox in bboxes:
                    added = recorded_bbox.check_n_add_bbox_instance(bbox_instance)
                if not added:
                    new_bbox = BoundingBox(self)
                    new_bbox.add_bbox_instance(bbox_instance, frame)
                    bboxes.append(new_bbox)
        to_skip = []
        final_bboxes_idx = []
        for i in range(len(bboxes)-1):
            if i in to_skip:
                continue
            for j in range(i+1, len(bboxes)):
                if j in to_skip:
                    continue
                merged = bboxes[i].merge_bounding_box(bboxes[j])
                if merged:
                    if i not in final_bboxes_idx:
                        final_bboxes_idx.append(i)
                    if j not in to_skip:
                        to_skip.append(j)
        if len(bboxes) > 0:
            last_idx = len(bboxes)-1
            if last_idx not in to_skip:
                final_bboxes_idx.append(last_idx)
        self.bboxes = [bboxes[x] for x in final_bboxes_idx]
        return self.bboxes

    def first_stage_classify_bboxes(self):
        for bbox in self.bboxes:
            bbox.check_static_type()
        if len(self.possible_channel) > 0 and self.channel_name is not None:
            for bbox in self.possible_channel:
                bbox.check_static_type(not_channel=True)
        elif len(self.possible_channel) > 0 and self.channel_name is None:
            for bbox in self.possible_channel:
                bbox.type = "channel"
                self.known_channels.append(bbox.bboxes[0].text.lower().strip())
            save_known_channels(self.known_channels, known_channels_path)

    def merge_bboxes(self):
        bbox_groups = []
        for bbox in self.bboxes:
            if bbox.consider_merging is True:
                # print(bbox.bboxes[0].text, bbox.overall_coords, bbox.prop_zero, bbox.overall_colours)
                added = False
                for bboxg in bbox_groups:
                    added = bboxg.checknadd_bbox(bbox)
                if not added:
                    bbox_groups.append(BoundingBoxGroup(bbox, self))
        to_skip = []
        final_bboxes_idx = []
        for i in range(len(bbox_groups)-1):
            if i in to_skip:
                continue
            for j in range(i+1, len(bbox_groups)):
                if j in to_skip:
                    continue
                merged = bbox_groups[i].merge_bboxg(bbox_groups[j])
                if merged:
                    if i not in final_bboxes_idx:
                        final_bboxes_idx.append(i)
                    if j not in to_skip:
                        to_skip.append(j)
        if len(bbox_groups) > 0:
            last_idx = len(bbox_groups)-1
            if last_idx not in to_skip:
                final_bboxes_idx.append(last_idx)
        self.bbox_groups = [bbox_groups[x] for x in final_bboxes_idx]
        for bboxg in self.bbox_groups:
            bboxg.checknupdate_rc()

    def save_to_video(self, out_dir):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #width of image
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 1 if sample_rate > 1 else int(1/sample_rate)
        out_file = '{}/{}.mp4'.format(out_dir, self.base_filename+"_bbox")
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        for frame in self.frames:
            frame.generate_bbox_list()
            for bbox in frame.bboxes:
                frame.image_w_bbox = plot_bbox(frame.image_w_bbox, str(bbox.type), get_plot_coords(bbox.overall_coords), (0,0,0))
            out.write(frame.image_w_bbox)
        out.release()
        cap.release()

    def check_subtitles(self):
        r = sr.Recognizer()
        # base_filename = self.video_path.split('/')[-1].split('.')[0]
        # Convert mp4 file to wav file
        audio_file = '{}/{}.wav'.format(self.audio_save_path, self.base_filename)
        assert self.video_path.endswith(".mp4")
        if not os.path.exists(audio_file):
            ffmpeg_command = 'ffmpeg -i "{}" -ac 1 -ar 16000 "{}"'.format(self.video_path, audio_file)
            call_ffmpeg = subprocess.Popen(ffmpeg_command, universal_newlines=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            f1 = call_ffmpeg.stdout.read()
            f2 = call_ffmpeg.wait()

        temp = sr.AudioFile(audio_file)
        with temp as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            temp_audio = r.record(source)

        if self.lang == 'eng':
            try:
                pred_text = r.recognize_google(temp_audio)#, language='en-SG')
                # pred_text = r.recognize_google_cloud(temp_audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            except sr.UnknownValueError:
                pred_text = ''
            joiner = ' '
        elif self.lang == 'ch':
            try:
                pred_text = r.recognize_google(temp_audio, language='zh')
                # pred_text = r.recognize_google_cloud(temp_audio, language='zh', credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            except sr.UnknownValueError:
                pred_text = ''
            joiner = ''

        # print('ASR', pred_text, '\n')
        lev_threshold = 500
        min_lev = 99999
        for bbox in self.bboxes:
            text = [sub_bbox.text for sub_bbox in bbox.bboxes]
            distinct_text = []
            for x in text:
                repeat = False
                for y in distinct_text:
                    if lev(x,y) < 5:
                        repeat = True
                        break
                if not repeat:
                    distinct_text.append(x)
            distinct_text = joiner.join(distinct_text)
            lev_dist = lev(pred_text, distinct_text)
            # print(distinct_text, lev_dist, '\n')
            if lev_dist < min_lev and lev_dist < lev_threshold:
                min_lev = lev_dist
                subtitle_info = [[sub_bbox.coords for sub_bbox in bbox.bboxes], distinct_text]
                bbox.type = 'subtitle'
        
        return subtitle_info
    
    def check_bounding_boxes_group(self):
        for i, bboxg in enumerate(self.bbox_groups):
            print(i, bboxg.texts[0])
            print(bboxg.coords)

    def compare_shots(self):
        shot_key_frame_detect(self.video_path, self.shot_path)
        shot_info = pd.read_csv('{}/{}.txt'.format(self.shot_path, self.base_filename), sep=' ', header=None)
        shots = glob('{}/{}_*.jpg'.format(shot_path, self.base_filename))
        shots.sort(key=natural_keys)
        shot_bbox_hist, _ = track_bboxes(shots,'shot')
        scene_text_bboxes = find_scene_text(shot_bbox_hist, shot_info)

        cap = cv2.VideoCapture(self.video_path)
        self.org_fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        for frame in self.frames:
            translated_frame_no = translate_frame(self.org_fps, frame.frame_no)
            for st_bbox in scene_text_bboxes:
                if translated_frame_no in list(range(st_bbox[1][0], st_bbox[1][1])):
                    # print(frame.frame_no, st_bbox)
                    for bbox in frame.bbox_instances:
                        bbox_coords = [bbox.coords[0][0], bbox.coords[0][1], bbox.coords[2][0], bbox.coords[2][1]]
                        if overlap(st_bbox[0], bbox_coords, 'shot') and bbox.bbox.type == None:
                            # _scene_text_bboxes.append([frame.frame_no, translated_frame_no, bbox_coords])
                            bbox.bbox.type = 'scene text'

    def check_no_audio_subtitle(self):
        # audio_file = '{}/{}.wav'.format(self.audio_save_path, self.base_filename)
        # sound = AudioSegment.from_file(audio_file, format="wav")
        # silent_chunks = detect_silence(sound, min_silence_len=1000, silence_thresh=-40)
        # # Convert to translated frame num 
        # silent_frames = [[int(start/sample_rate), int(stop/sample_rate)+1] for start, stop in silent_chunks]
        # filtered_silent_frames = []
        # for i, frames in enumerate(silent_frames):
        #     if i == 0:
        #         filtered_silent_frames.append(frames)
        #         prev_frames = frames
        #         continue
        #     if prev_frames[1] > frames[0]:
        #         filtered_silent_frames[len(filtered_silent_frames)-1][1] = frames[1]
        #     else:
        #         filtered_silent_frames.append(frames)
        #     prev_frames = frames
        
        # subtitle_bboxes = []
        # for silence in filtered_silent_frames:
        #     frame_bboxes = []
        #     for idx in range(silence[0], silence[-1]):
        #         frame_bboxes.append([(bbox.coords, (bbox.text, None)) for bbox in self.frames[idx].bbox_instances])
        #     bbox_hist, _ = track_bboxes(frame_bboxes, 'no_audio', x_margin=200, y_margin=300, lead=silence[0])
            
        #     # print('BBOX HIST', bbox_hist)
        #     for bbox in bbox_hist:
        #         frame_range = bbox[1]
        #         num_frames = frame_range[-1] - frame_range[0]
        #         # Detect subtitle bbox
        #         find_subtitle_bbox(subtitle_bboxes, num_frames, bbox, self.height)

        subtitle_bboxes = []
        frame_bboxes = []
        for frame in self.frames:
            frame_bboxes.append([(bbox.coords, (bbox.text, None)) for bbox in frame.bbox_instances])
        bbox_hist, _ = track_bboxes(frame_bboxes, 'no_audio')#, x_margin=200, y_margin=300)
        
        # print('BBOX HIST', bbox_hist)
        for bbox in bbox_hist:
            frame_range = bbox[1]
            num_frames = frame_range[-1] - frame_range[0]
            # Detect subtitle bbox
            find_subtitle_bbox(subtitle_bboxes, num_frames, bbox, self.height)

        # print('NO AUDIO SUB', subtitle_bboxes)

        for frame in self.frames:
            for sub_bbox in subtitle_bboxes:
                if frame.frame_no in list(range(sub_bbox[1][0], sub_bbox[1][1])):
                    for bbox in frame.bbox_instances:
                        bbox_coords = [bbox.coords[0][0], bbox.coords[0][1], bbox.coords[2][0], bbox.coords[2][1]]
                        # print(sub_bbox[-1], sub_bbox[0], bbox_coords)
                        if overlap(sub_bbox[0], bbox_coords, 'no_audio'):#, x_margin=100, y_margin=100):
                            # print('YES')
                            bbox.bbox.type = 'subtitle'

        # print('BEFORE', silent_frames)
        # print('AFTER', filtered_silent_frames)
        # for silence in filtered_silent_frames:
        #     for bbox in self.bboxes:
        #         bboxes_height = [sub_bbox.coords[2][-1] for sub_bbox in bbox.bboxes]
        #         frames = [sub_bbox.frame.frame_no for sub_bbox in bbox.bboxes]
        #         num_frames = max(frames) - min(frames)
        #         in_ms = num_frames * sample_rate
        #         if silence[0] >= min(frames) and silence[-1] <= max(frames):
        #             print(frames, bbox.texts)
        #             if sum(bboxes_height)/len(bboxes_height) <= int(self.height/2):
        #                 bbox.type = 'subtitle'
    
    def none_to_st(self):
        for frame in self.frames:
            for bbox in frame.bbox_instances:
                if bbox.bbox.type == None:
                    bbox.bbox.type = 'scene text'

def run(video_path, audio_save_path, shot_path, out_dir): 
    if os.path.isfile(video_path):
        video_paths = [video_path]
    else:
        video_paths = [os.path.join(video_path, x) for x in os.listdir(video_path)]

    for video_path in video_paths:
        # out_dir = "/home/hcari/trg/visualize"
        video = Video(video_path, audio_save_path, shot_path)
        video.generate_bboxes()
        video.first_stage_classify_bboxes()
        video.merge_bboxes()
        video.check_subtitles()
        video.compare_shots()
        video.check_no_audio_subtitle()
        video.none_to_st()
        # video.check_bounding_boxes_group()
        video.save_to_video(out_dir)

# video_path = "/home/hcari/trg/videos/"
# audio_save_path = "/home/hcari/trg/visualize/wav_files"
video_path = '../montage_detection/sinovac_ver/query_video/ChineseSinovacVideo.mp4'
audio_save_path = '../non_commit_trg/audio'
shot_path = '../montage_detection/images/sinovac/query'
out_dir = '../non_commit_trg/output'

# video_paths = ["/home/hcari/trg/videos/First day of Sinovac vaccine roll-out – one clinic shares experience _ THE BIG STORY [MucpzHvMKkw].mp4"]
run(video_path, audio_save_path, shot_path, out_dir)