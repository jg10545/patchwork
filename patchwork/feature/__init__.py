from patchwork.feature._contextencoder import build_inpainting_network, ContextEncoderTrainer
from patchwork.feature._models import BNAlexNetFCN
from patchwork.feature._deepcluster import DeepClusterTrainer
from patchwork.feature._simclr import SimCLRTrainer
from patchwork.feature._autoencoder import AutoEncoderTrainer
from patchwork.feature._multitask import MultiTaskTrainer
from patchwork.feature._moco import MomentumContrastTrainer
from patchwork.feature._byol import BYOLTrainer