# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp

data_file = './data/ActivityNet'
video_list = f'{data_file}/video_info_new.csv'
anno_file = f'{data_file}/anet_anno_action.json'
rawframe_dir = f'{data_file}/rawframes'
action_name_list = './tools/data/activitynet/action_name.csv' # Two in our case

# Rawframes are in the same folder but there are sets for frame : flowx : flowy
train_rawframe_dir = rawframe_dir
val_rawframe_dir = rawframe_dir

# The dataset labeling
json_file = f'{data_file}/activity_net.v1-3.min.json'


def generate_rawframes_filelist():
    # get the list of files
    # Load the .json file
    load_dict = json.load(open(json_file))
    # other file with the same information I guess
    anet_labels = open(action_name_list).readlines()
    anet_labels = [x.strip() for x in anet_labels[1:]] # skip the first one, as it is the csv file and it is just the category name

    train_dir_list = [
        osp.join(train_rawframe_dir, x) for x in os.listdir(train_rawframe_dir) # get filenames for training part
    ]
    val_dir_list = [
        osp.join(val_rawframe_dir, x) for x in os.listdir(val_rawframe_dir) # get frames for the validation
    ]

    def simple_label(anno):
        label = anno[0]['label'] # extract the label and convert it to the index: in our case that just 0 or 1
        return anet_labels.index(label)

    def count_frames(dir_list, video):
        for dir_name in dir_list: # go trough all the dirs
            if video in dir_name: # and then vides???
                #print(f"Counting files from: {dir_name}")
                files = os.listdir(dir_name)
                img_files_num = len([f for f in files if f.startswith('img_')])
                flowx_files_num = len([f for f in files if f.startswith('flow_x')])
                flowy_files_num = len([f for f in files if f.startswith('flow_y')])
                num_frames = min([img_files_num, flowx_files_num, flowy_files_num])
                return osp.basename(dir_name), num_frames
        return None, None

    database = load_dict['database']
    training = {}
    validation = {}
    key_dict = {}

    for k in database:
        data = database[k]
        subset = data['subset']

        if subset in ['training', 'validation']:
            annotations = data['annotations']
            label = simple_label(annotations)
            if subset == 'training':
                dir_list = train_dir_list
                data_dict = training
            else:
                dir_list = val_dir_list
                data_dict = validation

        else:
            continue

        gt_dir_name, num_frames = count_frames(dir_list, k)
        if gt_dir_name is None:
            continue
        data_dict[gt_dir_name] = [num_frames, label]
        key_dict[gt_dir_name] = k

    train_lines = [
        k + ' ' + str(training[k][0]) + ' ' + str(training[k][1])
        for k in training
    ]
    val_lines = [
        k + ' ' + str(validation[k][0]) + ' ' + str(validation[k][1])
        for k in validation
    ]

    with open(osp.join(data_file, 'anet_train_video.txt'), 'w') as fout:
        fout.write('\n'.join(train_lines))
    with open(osp.join(data_file, 'anet_val_video.txt'), 'w') as fout:
        fout.write('\n'.join(val_lines))

    def clip_list(k, anno, video_anno):
        duration = anno['duration']
        num_frames = video_anno[0]
        fps = num_frames / duration
        segs = anno['annotations']
        lines = []
        for seg in segs:
            segment = seg['segment']
            label = seg['label']
            label = anet_labels.index(label)
            start, end = int(segment[0] * fps), int(segment[1] * fps)
            if end > num_frames - 1:
                end = num_frames - 1
            newline = f'{k} {start} {end - start + 1} {label}'
            lines.append(newline)
        return lines

    train_clips, val_clips = [], []
    for k in training:
        train_clips.extend(clip_list(k, database[key_dict[k]], training[k]))
    for k in validation:
        val_clips.extend(clip_list(k, database[key_dict[k]], validation[k]))

    with open(osp.join(data_file, 'anet_train_clip.txt'), 'w') as fout:
        fout.write('\n'.join(train_clips))
    with open(osp.join(data_file, 'anet_val_clip.txt'), 'w') as fout:
        fout.write('\n'.join(val_clips))


if __name__ == '__main__':
    generate_rawframes_filelist()
