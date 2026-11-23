"""Pixel-level evaluation using raw class IDs rather than colorized visualizations."""
from __future__ import annotations
import numpy as np
import torch

def confusion_matrix(pred:torch.Tensor,target:torch.Tensor,num_classes:int,ignore_index=255)->np.ndarray:
    pred=pred.detach().cpu().numpy().astype("int64").ravel()
    target=target.detach().cpu().numpy().astype("int64").ravel()
    valid=(target!=ignore_index)&(target>=0)&(target<num_classes)
    valid&=(pred>=0)&(pred<num_classes)
    count=np.bincount(num_classes*target[valid]+pred[valid],minlength=num_classes**2)
    return count.reshape(num_classes,num_classes)

def summarize(cm:np.ndarray)->dict:
    tp=cm.diagonal()
    union=cm.sum(axis=1)+cm.sum(axis=0)-tp
    iou=np.full(len(tp),np.nan,dtype=float)
    present=union>0
    iou[present]=tp[present]/union[present]
    valid_count=int(cm.sum())
    return {"pixel_accuracy":float(tp.sum()/valid_count) if valid_count else None,
            "mean_iou":float(np.nanmean(iou)) if present.any() else None,
            "per_class_iou":[float(i) if np.isfinite(i) else None for i in iou],
            "confusion_matrix":cm.tolist(),"evaluated_pixels":valid_count}
