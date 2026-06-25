from utils.dataset import ChangeDetectionDataset
from utils.func import clip_gradient
from utils.metric_tool import AvgMeter, ConfuseMatrixMeter, BCEDICE_loss
from utils.lr_scheduler import get_scheduler