


import argparse
import yaml
from munch import Munch
import glob,tqdm
import os.path as osp
import numpy as np

import torch

from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.data.svg3 import SVGDataset
from svgnet.util  import get_root_logger, load_checkpoint
from svgnet.evaluation import PointWiseEval,InstanceEval

import time

def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--datadir", type=str, help="the path to dataset")
    parser.add_argument("--out", type=str, help="the path to save results")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    logger = get_root_logger()

    model = svgnet(cfg.model).cuda()
   
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)
    data_list = glob.glob(osp.join(args.datadir,"*_s2.json"))
    logger.info(f"Load dataset: {len(data_list)} svg")
    
    sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes,ignore_label=cfg.model.semantic_classes,gpu_num=1)
    instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes,ignore_label=cfg.model.semantic_classes,gpu_num=1)
    save_dicts = []
    total_times = []
    with torch.no_grad():
        model.eval()
        for svg_file in tqdm.tqdm(data_list):
            coords, feats, labels,lengths,layerIds = SVGDataset.load(svg_file,idx=1)
            coords -= np.mean(coords, 0)
            offset = [coords.shape[0]]
            layerIds = torch.LongTensor(layerIds)
            offset = torch.IntTensor(offset)
            coords,feats,labels = torch.FloatTensor(coords), torch.FloatTensor(feats), torch.LongTensor(labels)
            batch = (coords,feats,labels,offset, torch.FloatTensor(lengths),layerIds)
            
            torch.cuda.empty_cache()
            
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                t1 = time.time()
                res = model(batch,return_loss=False)
                t2 = time.time()
                total_times.append(t2-t1)
                sem_preds = torch.argmax(res["semantic_scores"],dim=1).cpu().numpy()
                sem_gts = res["semantic_labels"].cpu().numpy()
                sem_point_eval.update(sem_preds, sem_gts)
                instance_eval.update(
                    res["instances"],
                    res["targets"],
                    res["lengths"],
                )
                save_dicts.append({
                    "filepath": svg_file.replace(".json",".svg"),
                    "sem": res["semantic_scores"].cpu().numpy(),
                    "ins": res["instances"],
                    "targets":res["targets"],
                    "lengths":res["lengths"],
                })
                    
                    
    np.save('sem_ins_split_val.npy', save_dicts)            
    logger.info("Evaluate semantic segmentation")
    sem_point_eval.get_eval(logger)
    logger.info("Evaluate panoptic segmentation")
    instance_eval.get_eval(logger)
 
if __name__ == "__main__":
    main()       
        