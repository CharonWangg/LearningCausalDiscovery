lr = dict(type='choice', range=[0.00005, 0.0001, 0.0005, 0.001])
train_batch_size = dict(type='choice', range=[128, 256, 512, 1024])
alpha = dict(type='choice', range=[0.1, 0.25, 0.5, 0.7, 0.9])
gamma = dict(type='choice', range=[2, 3, 4, 5, 7, 9])
patch_size = dict(type='choice', range=[32, 64, 128, 256])

n_trials = 1
monitor = {"AUPRC": "val_average_precision"}
