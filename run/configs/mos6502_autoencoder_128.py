# model settings
num_classes = 2
loss = [dict(type="TorchLoss", loss_name="MSELoss", loss_weight=1.0)]

model = dict(
    arch=dict(
            type="Autoencoder",
            in_channels=2,
            hidden_size=128,
            mlp_ratio=1,
            input_length=30*128,
            embedding_size=128,
            patch_size=32,
            rec_error_threshold=0.06,
            num_classes=num_classes,
            losses=loss
    ),
    evaluation=dict(
        metrics=[
            dict(
                type='TorchMetrics',
                metric_name='AveragePrecision',
                prob=True, task='binary'
            ),
            dict(
                type="TorchMetrics",
                metric_name="AUROC",  # step_reduction='mean',
                prob=True, task='binary'
            ),
        ]
    ),
)

# dataset settings
dataset_type = "NMOS6502"

interval = 1

data = dict(
    train_batch_size=1024,  # for single card
    val_batch_size=2048,
    test_batch_size=2048,
    num_workers=8,
    train=dict(
        type="NMOS6502Neg",
        data_root='.cache/sim_data/DonkeyKong/HR/window_0_128/Regular_3510_step_128_rec_30_window_0_128.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_0_128/csv/fold_seed_42/train_ds_3.0_unique_True.csv",
        interval=interval,
        only_neg=True,
        shift_range=(-900, 900),
    ),
    val=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_128_256/Regular_3510_step_128_rec_30_window_128_256.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_128_256/csv/fold_seed_42/val_ds_1.0_unique_True.csv",
        interval=interval,
    ),
    test=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_384_512/Regular_3510_step_128_rec_30_window_384_512.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_384_512/csv/fold_seed_42/test_ds_1.0_unique_True.csv",
        interval=interval,
    ),
)

log = dict(
    project_name="nmos6502_official",
    work_dir="work_dir",
    exp_name="mos6502_autoencoder_128",
    logger_interval=50,
    monitor="val_average_precision",
    logger=True,  # use wandb logger
    checkpoint=dict(
        type="ModelCheckpoint",
        top_k=1,
        mode="max",
        verbose=True,
        save_last=True,
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
    max_iters=50,
    optimizer=dict(type="AdamW",
                   lr=1e-3,
                   weight_decay=0.05),
    scheduler=dict(
        min_lr=0.0,
    ),
)
