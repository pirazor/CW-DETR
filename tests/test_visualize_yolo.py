from __future__ import annotations

import torch

from cwdetr.engine.visualize_yolo import _ground_truth_rows, _prediction_rows


def test_prediction_rows_scale_model_boxes_back_to_original_image():
    logits = torch.full((1, 2, 3), -20.0)
    logits[0, 0, 2] = 20.0
    det_out = {
        "pred_logits": logits,
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.25, 0.50],
                                     [0.1, 0.1, 0.1, 0.1]]]),
    }
    rows = _prediction_rows(
        det_out, image_hw=(384, 640), orig_hw=(768, 1280),
        class_names=["car", "person", "sign"], score_thresh=0.5, topk=10)
    assert len(rows) == 1
    box, name, score, kind = rows[0]
    assert name == "sign"
    assert kind == "pred"
    assert score > 0.99
    assert torch.allclose(torch.tensor(box), torch.tensor([480.0, 192.0, 800.0, 576.0]))


def test_ground_truth_rows_use_original_image_size():
    target = {
        "labels": torch.tensor([1]),
        "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.50]]),
    }
    rows = _ground_truth_rows(target, orig_hw=(768, 1280),
                              class_names=["car", "person"])
    assert rows == [([480.0, 192.0, 800.0, 576.0], "person", 1.0, "gt")]
