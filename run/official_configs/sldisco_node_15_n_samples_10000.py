n_nodes = 15
loss = [
    dict(
        type='TorchLoss',
        loss_name='BCEWithLogitsLoss',
        multi_label=True,
        to_float=True,
        loss_weight=1.0)
]
model = dict(
    type='SLDiscoEncoder',
    backbone=dict(input_length=15),
    head=dict(
        channels=128,
        losses=[
            dict(
                type='TorchLoss',
                loss_name='BCEWithLogitsLoss',
                multi_label=True,
                to_float=True,
                loss_weight=1.0)
        ]),
    evaluation=dict(
        metrics=[dict(type='TorchMetrics', metric_name='Accuracy', prob=True)
                 ]))
dataset_type = 'CDLiNGAM'
train_pipeline = [dict(type='ToTensor')]
test_pipeline = [dict(type='ToTensor')]
data = dict(
    train_batch_size=1024,
    val_batch_size=6196,
    test_batch_size=6196,
    num_workers=4,
    train=dict(
        type=dataset_type,
        data_root='.cache/lingam/train',
        online=True,
        n_samples=100000,
        time_steps=100,
        adj_cfg=dict(
            markov_class=True,
            n_nodes=15,
            density=(0.0, 0.8),
            low=0.1,
            high=2.0,
            allow_negative=True),
        noise_cfg=dict(gaussian=True, scale=(0.5, 2.0), permutate=True),
        to_corr=True,
        sampler=None,
        pipeline=[dict(type='ToTensor')]),
    val=dict(
        type=dataset_type,
        data_root='.cache/lingam/val',
        online=True,
        n_samples=5000,
        time_steps=100,
        adj_cfg=dict(
            markov_class=False,
            n_nodes=15,
            density=(0.0, 0.8),
            low=0.1,
            high=2.0,
            allow_negative=True),
        noise_cfg=dict(gaussian=True, scale=(0.5, 2.0), permutate=True),
        to_corr=True,
        sampler='SequentialSampler',
        pipeline=[dict(type='ToTensor')]),
    test=dict(
        type=dataset_type,
        data_root='.cache/lingam/test',
        online=True,
        n_samples=5000,
        time_steps=100,
        adj_cfg=dict(
            markov_class=False,
            n_nodes=15,
            density=(0.0, 0.8),
            low=0.1,
            high=2.0,
            allow_negative=True),
        noise_cfg=dict(gaussian=True, scale=(0.5, 2.0), permutate=True),
        to_corr=True,
        sampler='SequentialSampler',
        pipeline=[dict(type='ToTensor')]))
log = dict(
    project_name='netsim',
    work_dir='work_dir',
    exp_name='sldisco_node_15_n_samples_10000',
    logger_interval=50,
    monitor='val_loss_epoch',
    logger=None,
    checkpoint=dict(
        type='ModelCheckpoint',
        top_k=1,
        mode='min',
        verbose=True,
        save_last=False),
    earlystopping=dict(
        mode='min',
        strict=False,
        patience=150,
        min_delta=0.0001,
        check_finite=True,
        verbose=True))
resume_from = None
cudnn_benchmark = True
optimization = dict(
    type='epoch',
    max_iters=150,
    optimizer=dict(
        type='AdamW', lr=0.01, betas=(0.9, 0.999), weight_decay=0.05),
    scheduler=dict(
        type='CosineAnnealing', interval='step', min_lr=1e-06, warmup=None))
