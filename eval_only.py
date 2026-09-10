from types import SimpleNamespace

from utils import get_loader
from models.evaluator import CDEvaluator


args = SimpleNamespace(

    gpu_ids=[0],

    project_name='ChangeFormer_LEVIR_TEST',

    checkpoint_root='checkpoints',

    vis_root='vis',

    num_workers=2,

    dataset='CDDataset',

    data_name='LEVIR',

    batch_size=1,

    split='test',

    img_size=256,

    n_class=2,

    embed_dim=256,

    net_G='ChangeFormerV6',

    checkpoint_dir=r'checkpoints\ChangeFormer_LEVIR_TEST',

    vis_dir=r'vis\ChangeFormer_LEVIR_TEST'
)


# Load test dataset only
dataloader = get_loader(
    data_name='LEVIR',
    img_size=256,
    batch_size=1,
    split='test',
    is_train=False,
    dataset='CDDataset'
)


# Create evaluator
model = CDEvaluator(
    args=args,
    dataloader=dataloader
)


# Evaluate existing best checkpoint
model.eval_models(
    checkpoint_name='best_ckpt.pt'
)