"""Unified ADAS detection taxonomy and per-dataset label maps.

A single 13-class space lets BDD100K and nuScenes train one detection head.
``traffic_sign`` is index 11 to match ``source_det_class`` in the configs — its
crops are routed to the fine-grained sign-classification head.
"""

CLASSES = [
    "car",                  # 0
    "truck",                # 1
    "bus",                  # 2
    "trailer",              # 3
    "construction_vehicle", # 4
    "pedestrian",           # 5
    "motorcycle",           # 6
    "bicycle",              # 7
    "rider",                # 8
    "traffic_cone",         # 9
    "barrier",              # 10
    "traffic_sign",         # 11
    "traffic_light",        # 12
]
NUM_CLASSES = len(CLASSES)
NAME_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# BDD100K 'category' string -> unified id
BDD100K_MAP = {
    "car": 0, "truck": 1, "bus": 2, "person": 5, "rider": 8,
    "motor": 6, "motorcycle": 6, "bike": 7, "bicycle": 7,
    "train": 2, "traffic sign": 11, "traffic light": 12, "trailer": 3,
}

# nuScenes detection name -> unified id
NUSCENES_MAP = {
    "vehicle.car": 0, "vehicle.truck": 1, "vehicle.bus.rigid": 2,
    "vehicle.bus.bendy": 2, "vehicle.trailer": 3, "vehicle.construction": 4,
    "human.pedestrian.adult": 5, "human.pedestrian.child": 5,
    "vehicle.motorcycle": 6, "vehicle.bicycle": 7,
    "movable_object.trafficcone": 9, "movable_object.barrier": 10,
}
