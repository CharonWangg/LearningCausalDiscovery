# model settings
num_classes = 2
loss = [dict(type="FocalLoss", loss_weight=1.0)]

model = dict(
    arch=dict(type="TCN",
              in_channels=2,
              num_classes=num_classes,
              input_length=1000,
              embedding_size=128,
              patch_size=24,
              stride=24,
              hidden_size=128,
              num_layers=2,
              input_norm=False,
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
                metric_name="AUROC",
                prob=True,
                pos_label=1,
            ),
            dict(
                type="TorchMetrics",
                metric_name="HammingDistance",
                prob=True,
                threshold=0.8,
            ),
        ]
    ),
)

# dataset settings
dataset_type = "Dream3"
train_dir = [
    ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli1.pt',
     '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli1.tsv'),
    ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size50Ecoli1.pt',
     '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize50-Ecoli1.tsv'),
    ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size10Ecoli1.pt',
     '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Ecoli1.tsv'),
    # ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast1.pt',
    #  '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize50-Yeast1.tsv'),
    # ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast2.pt',
    #  '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize50-Yeast2.tsv'),
]

val_dir = [
    ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli2.pt',
     '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv'),
    # ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast3.pt',
    #  '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize50-Yeast3.tsv')
]

test_dir = [
    ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli2.pt',
     '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv'),
    # ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast3.pt',
    #  '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize50-Yeast3.tsv')
]


data = dict(
    train_batch_size=256,  # for single card
    val_batch_size=12284,
    test_batch_size=12284,
    num_workers=0,
    train=dict(
        type=dataset_type,
        multiple=True,
        multiple_key="data_root",
        data_root=train_dir,
        percentage=1.0,
    ),
    val=dict(
        type=dataset_type,
        multiple=True,
        multiple_key="data_root",
        data_root=val_dir,
        percentage=1.0,
    ),
    test=dict(
        type=dataset_type,
        multiple=True,
        multiple_key="data_root",
        data_root=test_dir,
        percentage=1.0,
    ),
)

# yapf:disable
log = dict(
    project_name="dream3",
    work_dir="work_dir",
    exp_name="dream3_partial_patch_transformer_128_50ep_cosine_adamw_1e-3lr_0.05wd",
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
    max_iters=50,
    optimizer=dict(type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    scheduler=dict(type="CosineAnnealing", interval="step", min_lr=0),
)
