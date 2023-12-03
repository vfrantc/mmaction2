ann_file_test = 'data/ActivityNet/anet_val_video.txt'
ann_file_train = 'data/ActivityNet/anet_train_video.txt'
ann_file_val = 'data/ActivityNet/anet_val_video.txt'
checkpoint_config = dict(interval=5)
data_root = 'data/ActivityNet/rawframes'
data_root_val = 'data/ActivityNet/rawframes'
dataset_type = 'RawframeDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = './models/tsn_r50_320p_1x1x8_50e_flow_activitynet_finetune.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
lr_config = dict(
    policy='step', step=[
        60,
        120,
    ])
model = dict(
    backbone=dict(
        depth=50,
        in_channels=2,
        norm_eval=False,
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        type='ResNet'),
    cls_head=dict(
        average_clips='score',
        backbone_name=None,
        num_segments=None,
        spatial_type='avg',
        temporal_type='avg',
        type='FeatureHead'),
    data_preprocessor=dict(
        format_shape='NCHW',
        mean=[
            128,
            128,
        ],
        std=[
            128,
            128,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=None,
    train_cfg=None,
    type='Recognizer2D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001))
optimizer = dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=50,
        gamma=0.1,
        milestones=[
            20,
            40,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/content/mmaction2/data/ActivityNet/anet_train_video.txt',
        data_prefix=dict(img='/content/mmaction2/data/ActivityNet/rawframes'),
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=[
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=25,
                test_mode=True,
                twice_sample=False,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        start_index=1,
        test_mode=True,
        type='RawframeDataset',
        with_offset=True),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        out_file_path='./features/flow_train_feat.pkl/total_feats.pkl',
        type='DumpResults'),
]
test_pipeline = [
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True,
        type='SampleFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='TenCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
total_epochs = 10
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/ActivityNet/anet_train_video.txt',
        data_prefix=dict(img='data/ActivityNet/rawframes'),
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=[
            dict(
                clip_len=1, frame_interval=1, num_clips=8,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        start_index=1,
        type='RawframeDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=1, frame_interval=1, num_clips=8, type='SampleFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/ActivityNet/anet_val_video.txt',
        data_prefix=dict(img='data/ActivityNet/rawframes'),
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        pipeline=[
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        start_index=1,
        test_mode=True,
        type='RawframeDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,
        type='SampleFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './features/flow_train_feat.pkl/work_dir'
workflow = [
    (
        'train',
        5,
    ),
]