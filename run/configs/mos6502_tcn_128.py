# model settings
num_classes = 2
loss = [dict(type="FocalLoss", alpha=0.7, gamma=3, loss_weight=1.0)]

model = dict(
    arch=dict(
            type="TCN",
            in_channels=2,
            hidden_size=128,
            input_length=40*256,
            embedding_size=128,
            patch_size=32,
            num_layers=2,
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

# dataset settings
dataset_type = "NMOS6502"

interval = 10

data = dict(
    train_batch_size=512,  # for single card
    val_batch_size=512,
    test_batch_size=512,
    num_workers=0,
    train=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_256_512/Regular_3510_step_256_rec_400_window_256_512.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_256_512/csv/fold_seed_42/train_ds_0.01.csv",
        interval=interval,
    ),
    val=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_512_768/Regular_3510_step_256_rec_400_window_512_768.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_512_768/csv/fold_seed_42/val_ds_0.01.csv",
        interval=interval,
    ),
    test=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_768_1024/Regular_3510_step_256_rec_400_window_768_1024.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_768_1024/csv/fold_seed_42/test_ds_0.01.csv",
        interval=interval,
    ),
)

log = dict(
    project_name="nmos6502_official",
    work_dir="work_dir",
    exp_name="mos6502_tcn_128",
    logger_interval=50,
    monitor="val_average_precision",
    logger=True,  # use wandb logger
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
    max_iters=10,
    optimizer=dict(type="AdamW",
                   lr=5e-5,
                   weight_decay=0.05),
    scheduler=dict(
        min_lr=0.0,
    ),
)
