"""Run inference using the freshly trained PyTorch checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .model import FRRNStyle

def predict(checkpoint:Path,image:Path,output:Path,device="cpu"):
    payload=torch.load(checkpoint,map_location=device,weights_only=True)
    model=FRRNStyle(int(payload["num_classes"]),int(payload["width"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    im=Image.open(image).convert("RGB")
    h,w=int(payload["input_height"]),int(payload["input_width"])
    data=np.asarray(im.resize((w,h),Image.Resampling.BILINEAR),dtype=np.uint8).copy()
    x=torch.from_numpy(data).permute(2,0,1).unsqueeze(0).float().to(device)/255.
    with torch.no_grad():
        pred=model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(pred,"L").save(output)
    print(f"Wrote raw per-pixel class-ID mask: {output}")
    return pred

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    predict(a.checkpoint,a.image,a.output)

if __name__=="__main__":main()
