from patchwork.feature._contextencoder import build_inpainting_network, ContextEncoderTrainer
#from patchwork.feature._models import BNAlexNetFCN
from patchwork.feature._deepcluster import DeepClusterTrainer
from patchwork.feature._simclr import SimCLRTrainer
from patchwork.feature._autoencoder import AutoEncoderTrainer
from patchwork.feature._multitask import MultiTaskTrainer
from patchwork.feature._moco import MomentumContrastTrainer
from patchwork.feature._byol import BYOLTrainer
from patchwork.feature._hcl import HCLTrainer
from patchwork.feature._detcon import DetConTrainer
from patchwork.feature._clip import CLIPTrainer
from patchwork.feature._simsiam import SimSiamTrainer

import patchwork.feature.models