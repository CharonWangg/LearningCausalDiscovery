# model settings
num_classes = 2
loss = [dict(type='FocalLoss', loss_weight=1.0)]

model = dict(
    type='BaseEncoderDecoder',
    backbone=dict(
        type='Transformer',
        embedding=dict(type='PatchEmbedding',
                       in_channels=2,
                       input_length=200,
                       embedding_size=128,
                       patch_size=24,
                       stride=12,
                       mode='1d',
                       input_norm=dict(type='BatchNorm1d',
                                       num_features=2,
                                       reshape_list=['b l c', 'b c l'])),
        hidden_size=128,
        num_layers=4,
        nhead=8,
    ),
    head=dict(
        type='BasePooler',
        pooler_type='mean',
        in_index=-1,
        in_channels=128,
        dropout=0.3,
        num_classes=num_classes,
        channels=None,
        losses=loss
    ),
    # auxiliary_head=None,
    evaluation=dict(metrics=[dict(type='TorchMetrics', metric_name='Accuracy',  # step_reduction='mean',
                                    prob=True),
                             dict(type='TorchMetrics', metric_name='AUROC',  # step_reduction='mean',
                                    prob=True, pos_label=1),
                             dict(type='TorchMetrics', metric_name='HammingDistance',
                                    prob=True, threshold=0.8)])
)

# dataset settings
dataset_type = 'NetSim'
train_dir = ['.cache/netsim/sim1.mat',
             '.cache/netsim/sim2.mat',
             '.cache/netsim/sim3.mat',
             '.cache/netsim/sim4.mat',
             '.cache/netsim/sim8.mat',
             ]

val_dir = [ 
            '.cache/netsim/sim1.mat',
             '.cache/netsim/sim2.mat',
             '.cache/netsim/sim3.mat',
             '.cache/netsim/sim8.mat',
             '.cache/netsim/sim10.mat',
             '.cache/netsim/sim11.mat',
             '.cache/netsim/sim12.mat',
             '.cache/netsim/sim13.mat',
             '.cache/netsim/sim4.mat',
             '.cache/netsim/sim14.mat',
             '.cache/netsim/sim15.mat',
             '.cache/netsim/sim16.mat',
             '.cache/netsim/sim17.mat',
             '.cache/netsim/sim18.mat',
             '.cache/netsim/sim21.mat'
           ]

test_dir = [
             '.cache/netsim/sim1.mat',
             '.cache/netsim/sim2.mat',
             '.cache/netsim/sim3.mat',
             '.cache/netsim/sim8.mat',
             '.cache/netsim/sim10.mat',
             '.cache/netsim/sim11.mat',
             '.cache/netsim/sim12.mat',
             '.cache/netsim/sim13.mat',
             '.cache/netsim/sim4.mat',
             '.cache/netsim/sim14.mat',
             '.cache/netsim/sim15.mat',
             '.cache/netsim/sim16.mat',
             '.cache/netsim/sim17.mat',
             '.cache/netsim/sim18.mat',
             '.cache/netsim/sim21.mat',
            ]

train_pipeline = [dict(type='ToTensor')]
test_pipeline = [dict(type='ToTensor')]

data = dict(
    train_batch_size=4096,  # for single card
    val_batch_size=12284,
    test_batch_size=12284,
    num_workers=4,
    train=dict(
        type=dataset_type,
        multiple=True,
        multiple_key='data_root',
        data_root=train_dir,
        percentage=0.6,
        sampler=None,
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        multiple=True,
        multiple_key='data_root',
        data_root=val_dir,
        percentage=[0.6, 0.8],
        sampler='SequentialSampler',
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        multiple=True,
        multiple_key='data_root',
        data_root=test_dir,
        percentage=[0.6, 1.0],
        sampler='SequentialSampler',
        pipeline=test_pipeline
    ),
)

# yapf:disable
log = dict(
    project_name='netsim',
    work_dir='work_dir',
    exp_name='netsim_partial_patch_transformer_128_50ep_cosine_adamw_1e-3lr_0.05wd',
    logger_interval=50,
    monitor='val_auroc',
    logger=None,
    checkpoint=dict(type='ModelCheckpoint',
                    top_k=1,
                    mode='max',
                    verbose=True,
                    save_last=False,
                    ),
    earlystopping=dict(
            mode='max',
            strict=False,
            patience=20,
            min_delta=0.0001,
            check_finite=True,
            verbose=True
    )

)

resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    type='epoch',
    max_iters=50,
    optimizer=dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05),
    scheduler=dict(type='CosineAnnealing',
                   interval='step',
                 min_lr=1e-6,
                 warmup=None)

)


