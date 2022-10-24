# model settings
num_classes = 2
loss = [dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0)]

model = dict(
    type='BaseEncoderDecoder',
    backbone=dict(
        type='LSTM',
        embedding=dict(type='PatchEmbedding',
                       in_channels=2,
                       input_length=51200,
                       embedding_size=128,
                       patch_size=512,
                       mode='1d'),
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
    ),
    head=dict(
        type='BasePooler',
        pooler_type='mean',
        in_index=-1,
        in_channels=256,
        dropout=0.3,
        num_classes=num_classes,
        channels=None,
        losses=loss
    ),
    # auxiliary_head=None,
    evaluation = dict(metrics=[dict(type='TorchMetrics', metric_name='Accuracy',# step_reduction='mean',
                                    prob=True),
                               dict(type='TorchMetrics', metric_name='AUROC',# step_reduction='mean',
                                    prob=True, pos_label=1)])
)

# dataset settings
dataset_type = 'NMOS6502'
data_root = '.cache/sim_data/DonkeyKong/'
seqs_dir = 'HR/Regular_3510_step_256_rec_2e3.npy'

train_pipeline = [dict(type='TsAug', transforms=[dict(type='AddNoise', scale=0.5, normalize=False, seed=42)]),
                  dict(type='ToTensor')]
test_pipeline = [dict(type='TsAug', transforms=[dict(type='AddNoise', scale=0.5, normalize=False, seed=42)]),
                 dict(type='ToTensor')]

data = dict(
    train_batch_size=256,  # for single card
    val_batch_size=256,
    test_batch_size=256,
    num_workers=4,
    train=dict(
        type=dataset_type,
        data_root=data_root + seqs_dir,
        split=data_root + 'csv/fold_seed_42/train_sim_grouped.csv',
        sampler=None,
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root + seqs_dir,
        split=data_root + 'csv/fold_seed_42/valid_sim_grouped.csv',
        sampler='SequentialSampler',
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root + seqs_dir,
        split=data_root + 'csv/fold_seed_42/test_sim_grouped.csv',
        sampler='SequentialSampler',
        pipeline=test_pipeline
    ),
)

log = dict(
    project_name='nmos6502_official',
    work_dir='work_dir',
    exp_name='patch_lstm_128_0.5noise',
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
                 warmup=dict(type='LinearWarmup', period=0.1))

)


