import torch
from pathlib import Path

def save_state(model,
               dirpath,
               name):

    # if path doesn't exist create it
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    model_savename = dirpath / name
    torch.save(obj = model.state_dict(),
               f = model_savename)
