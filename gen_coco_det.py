import os
import json
import numpy as np
from collections import defaultdict

import torch, torchvision
#from svgnet.evaluation.point_wise_eval import LABELS
from pycocotools.coco import COCO
from coco_eval import CocoEvaluator
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
SVG_CATEGORIES = [
    #1-6 doors
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
    #7-10 window
    {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
    #11-27: furniture
    {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
    #28:stairs
    {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
    #29-30: equipment
    {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},

    #31-35: uncountable
    {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
    {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
]

def process_dt(input):
    json_file, instances, vis_dir = input
    det_dicts = defaultdict(list)
    filename = os.path.basename(json_file).replace('_s2.json', '.svg')
    data = json.load(open(json_file))
    png_path = os.path.join(vis_dir,filename.replace('.svg', '.png'))
    #if not os.path.exists(png_path): return
    image = Image.open(png_path)
    draw = ImageDraw.Draw(image,'RGBA')
    args = np.array(data["args"])
    num = args.shape[0]
    for instance in instances:
        mask, label, score = instance['masks'], instance['labels'], instance['scores']
        #if score<0.5: continue
        if not sum(mask): continue
        if label !=4 and sum(mask)<2: continue
        mask = np.array(mask).astype(np.bool_)[:num]
        arg = args[mask].reshape(-1,2) 
        if label==0 and sum(mask)<4: continue
        x1, y1 = np.min(arg[:,0],axis=0), np.min(arg[:,1],axis=0)
        x2, y2 = np.max(arg[:,0],axis=0), np.max(arg[:,1],axis=0)
        
        det_dicts[filename].append([x1,y1,x2,y2,score,label])
        if label>=30: continue
        #vis
        color = SVG_CATEGORIES[int(label)]["color"]
        #draw.rectangle([x1*7,y1*7,x2*7,y2*7],fill=None,outline=tuple(color),width=2)
        draw.rectangle([x1*7,y1*7,x2*7,y2*7],fill=tuple(color+[32]),width=2)
        #label = SVG_CATEGORIES[int(label)]["name"]
        text = '{}:{:.2f}'.format(str(label),score)
        draw.text((x1*7,y1*7),text,align='right',fill=(0,0,0))
    
    save = png_path.replace(".png","_res.png")
    image.save(save)   
    
    if det_dicts[filename]:
        preds = torch.tensor(det_dicts[filename])
        inds = torchvision.ops.boxes.batched_nms(preds[:,:4],
                    preds[:,4],
                    preds[:,5],
                    0.3)
        det_dicts[filename] = preds[inds]
    
    return det_dicts
   
if __name__ == "__main__":
    
    coco_res_npyfile = "coco_res_val.npy"
    vis_dir = "./spv2-norgb-val"
    detections = np.load(coco_res_npyfile,allow_pickle=True)
    import tqdm
    inputs = []
    for det in tqdm.tqdm(detections):
        svg_file = det['filepath']
        json_file = svg_file.replace('.svg','.json')
        instances = det['instances']
        inputs.append([json_file, instances, vis_dir])
    
    import mmcv
    det_dicts = mmcv.track_parallel_progress(process_dt,inputs,32)
    results = {}
    for det in det_dicts:
        results.update(det)
    ann_file = './val.cocojson'
    coco_gt = COCO(ann_file)
    coco_evaluator = CocoEvaluator(coco_gt,iou_types=['bbox'], useCats=True)
    
    for img_id in tqdm.tqdm(coco_gt.getImgIds()):
        img_info = coco_gt.loadImgs(img_id)
        image_id, image_path = img_info[0]['id'],img_info[0]['file_name']
        if image_path not in results.keys(): continue
        dets = results[image_path]
        if not len(dets): continue
        
        outs = {image_id: {
                    'boxes':dets[:,:4],
                    'scores':dets[:,4],
                    'labels':dets[:,-1],
                    }}
        coco_evaluator.update(outs)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize(ap50=True)
    
