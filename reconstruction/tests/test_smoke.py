from pathlib import Path
import numpy as np
from PIL import Image
import torch
from reproduction.data import make_synthetic,ImageMaskDataset
from reproduction.model import FRRNStyle
from reproduction.train import train
from reproduction.predict import predict
from reproduction.metrics import confusion_matrix,summarize

def test_end_to_end(tmp_path:Path):
    root=tmp_path/"synthetic"
    make_synthetic(root,4,2,height=32,width=32)
    ds=ImageMaskDataset(root,"train",(32,32),4)
    assert len(ds)==4
    im,gt=ds[0]
    assert im.shape==(3,32,32) and gt.shape==(32,32)
    model=FRRNStyle(4,8)
    with torch.no_grad():
        assert model(im.unsqueeze(0)).shape==(1,4,32,32)
    history=train(root,tmp_path/"results",epochs=1,height=32,width=32,num_classes=4,model_width=8,batch_size=2)
    assert len(history)==1 and history[0]["training_loss"]>0
    output=tmp_path/"prediction.png"
    labels=predict(tmp_path/"results/model.pt",root/"val/images/0000.png",output)
    assert output.is_file() and labels.shape==(32,32)
    assert set(np.unique(labels)).issubset({0,1,2,3})
    assert np.asarray(Image.open(output)).shape==(32,32)

def test_metrics_exact():
    target=torch.tensor([[0,0,1],[1,2,255]])
    pred=torch.tensor([[0,1,1],[1,2,3]])
    metrics=summarize(confusion_matrix(pred,target,4))
    assert metrics["evaluated_pixels"]==5
    assert abs(metrics["pixel_accuracy"]-0.8)<1e-9
    assert abs(metrics["per_class_iou"][0]-0.5)<1e-9

def test_invalid_raw_label_raises(tmp_path):
    root=tmp_path/"data"
    make_synthetic(root,1,1,height=32,width=32)
    Image.fromarray(np.full((32,32),19,dtype=np.uint8),"L").save(root/"train/masks/0000.png")
    ds=ImageMaskDataset(root,"train",(32,32),4)
    try:ds[0]
    except ValueError as e:assert "remap class IDs" in str(e)
    else:raise AssertionError("expected label validation error")
