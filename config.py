import torch

class Config:
    # Training
    batch_size = 128
    epochs = 10
    data_dir = './data'
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # HANS params (not all are used directly; for reference)
    hans_lr = 0.001
    hans_betas = (0.9, 0.999)
    hans_eps = 1e-8
    hans_weight_decay = 1e-4

    # 'cnn' or 'resnet'
    architecture = 'resnet'

    # Debug flags
    debug_mode = False
    log_grad_norms = True
    overfit_single_batch = False
