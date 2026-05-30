"""CW-DETR — ConnectedWise multi-task ADAS perception.

A single DINOv3 backbone + shared deformable-DETR decoder feeding five task
heads (detection, query-based tracking, lane/drivable segmentation, traffic-sign
sub-classification, and multimodal trajectory prediction), built on the RF-DETR
real-time detection lineage and tuned for Jetson Orin Nano / NX.

Public entry points:
    from cwdetr import build_cwdetr, load_config
    model = build_cwdetr(load_config("configs/cwdetr_nano_orin.yaml"))
"""
from cwdetr.config import CWDETRConfig, load_config
from cwdetr.models.cwdetr import CWDETR, build_cwdetr

__all__ = ["CWDETRConfig", "load_config", "CWDETR", "build_cwdetr"]
__version__ = "0.1.0"
