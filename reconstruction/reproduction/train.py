"""Train and evaluate the newly reconstructed PyTorch FRRN-style model."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import ImageMaskDataset
from .model import FRRNStyle
from .metrics import confusion_matrix,summarize

def train(root:Path,output:Path,epochs:int=1,height:int=32,width:int=64,
          num_classes:int=4,model_width:int=8,seed:int=42,device:str="cpu",
          batch_size:int=2):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    try:torch.set_num_threads(min(2,torch.get_num_threads()))
    except RuntimeError:pass
    target_device=torch.device(device)
    train_ds=ImageMaskDataset(root,"train",(height,width),num_classes)
    val_ds=ImageMaskDataset(root,"val",(height,width),num_classes)
    loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model=FRRNStyle(num_classes,model_width).to(target_device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss(ignore_index=255)
    history=[]
    for ep in range(epochs):
        model.train();total=0.;batches=0
        for img,gt in loader:
            img,gt=img.to(target_device),gt.to(target_device)
            optimizer.zero_grad()
            loss=criterion(model(img),gt)
            loss.backward();optimizer.step()
            total+=loss.item();batches+=1
        model.eval();cm=np.zeros((num_classes,num_classes),dtype=np.int64)
        with torch.no_grad():
            for img,gt in val_loader:
                p=model(img.to(target_device)).argmax(dim=1)
                cm+=confusion_matrix(p,gt,num_classes)
        metrics=summarize(cm)
        history.append({"epoch":ep+1,"training_loss":total/batches,"validation":metrics})
        print(f'epoch={ep+1} loss={total/batches:.4f} val_mIoU={metrics["mean_iou"]:.4f}')
    output.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"num_classes":num_classes,"width":model_width,
                "input_height":height,"input_width":width,"seed":seed,
                "note":"New PyTorch FRRN-style reproduction; NOT original TensorFlow checkpoint"},
               output/"model.pt")
    (output/"history.json").write_text(json.dumps(history,indent=2))
    return history

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("artifacts/run"))
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--height",type=int,default=32)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--classes",type=int,default=4)
    p.add_argument("--model-width",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    train(a.data,a.output,a.epochs,a.height,a.width,a.classes,a.model_width,device=a.device,batch_size=a.batch_size)

if __name__=="__main__":main()
