# model settings
num_classes = 2
loss = [dict(type="FocalLoss", loss_weight=1.0)]

model = dict(
    arch=dict(
            type="Transformer",
            in_channels=2,
            hidden_size=128,
            input_length=512,
            embedding_size=128,
            patch_size=8,
            num_layers=4,
            nhead=8,
            num_classes=num_classes,
            losses=loss
    ),
    evaluation=dict(
        metrics=[
            dict(
                type='TorchMetrics',
                metric_name='AveragePrecision',
                prob=True, num_classes=2
            ),
            dict(
                type="TorchMetrics",
                metric_name="AUROC",  # step_reduction='mean',
                prob=True,
                pos_label=1,
            ),
        ]
    ),
)

# dataset settings
dataset_type = "NMOS6502"
data_root = ".cache/sim_data/DonkeyKong/"
seqs_dir = "HR/Regular_3510_step_256_rec_2e3.npy"

data = dict(
    train_batch_size=256,  # for single card
    val_batch_size=256,
    test_batch_size=256,
    num_workers=4,
    train=dict(
        type=dataset_type,
        data_root=data_root + seqs_dir,
        split=data_root + "csv/fold_seed_42/train_sim_grouped.csv",
        sampler=None,
        interval=1000,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root + seqs_dir,
        split=data_root + "csv/fold_seed_42/valid_sim_grouped.csv",
        sampler="SequentialSampler",
        interval=1000,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root + seqs_dir,
        split=data_root + "csv/fold_seed_42/test_sim_grouped.csv",
        sampler="SequentialSampler",
        interval=1000,
    ),
)

log = dict(
    project_name="nmos6502_official",
    work_dir="work_dir",
    exp_name="patch_transformer_128_fold_seed_42",
    logger_interval=50,
    monitor="val_auroc",
    logger=None,
    checkpoint=dict(
        type="ModelCheckpoint",
        top_k=1,
        mode="max",
        verbose=True,
        save_last=False,
    ),
    earlystopping=dict(
        mode="max",
        strict=False,
        patience=5,
        min_delta=0.0001,
        check_finite=True,
        verbose=True,
    ),
)

resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    type="epoch",
    max_iters=20,
    optimizer=dict(type="AdamW",
                   lr=1e-3,
                   weight_decay=0.05),
    scheduler=dict(
        min_lr=0.0,
    ),
)
