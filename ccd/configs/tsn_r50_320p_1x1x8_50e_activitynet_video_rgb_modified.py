# RGB MODIFIED
# /content/mmaction2/configs/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_modified.py


_base_ = [
    '/content/mmaction2/configs/_base_/models/tsn_r50.py', '/content/mmaction2/configs/_base_/schedules/sgd_50e.py',
    '/content/mmaction2/configs/_base_/default_runtime.py'
]
# model settings
model = dict(cls_head=dict(num_classes=2, dropout_ratio=0.8))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ActivityNet/rawframes'
data_root_val = 'data/ActivityNet/rawframes'
ann_file_train = 'data/ActivityNet/anet_train_video.txt'
ann_file_val = 'data/ActivityNet/anet_val_video.txt'
ann_file_test = 'data/ActivityNet/anet_val_video.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDataset'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
    # Removed 'Collect' and 'ToTensor' to align with the working config
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
    # Removed 'Collect' and 'ToTensor' to align with the working config
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
    # Removed 'Collect' and 'ToTensor' to align with the working config
]

train_dataloader = dict(
    batch_size=8,  # changed from videos_per_gpu
    num_workers=2,  # changed from workers_per_gpu
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,  # assuming this is correct for your val set
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,  # from test_dataloader: videos_per_gpu
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric', metrics=['top_k_accuracy', 'mean_class_accuracy'])
test_evaluator = val_evaluator

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# runtime settings
work_dir = '/content/drive/MyDrive/3_Research_Replication/MMAction_V1/Our_Data/TSN_Files/rgb_checkpoints/'
load_from = ('https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth')
# load_from = '/content/mmaction2/data/ActivityNet/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth'