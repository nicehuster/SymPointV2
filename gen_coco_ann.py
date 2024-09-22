
import json
import numpy as np
from collections import defaultdict

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
LABELS = [x["name"] for x in SVG_CATEGORIES]



def process_gt(input):
    svg_path,json_path = input
    data = json.load(open(json_path))
    inst_labels = np.array(data["instanceIds"])
    sem_labels = np.array(data["semanticIds"])
    labels = np.concatenate([sem_labels[:,None],
                             inst_labels[:,None]],axis=1)
    uni_labels = np.unique(labels,axis=0)
    args = np.array(data["args"])
    det_dicts = defaultdict(list)
    filename = svg_path.split("/")[-1]
    w, h = data["width"], data['height']
    det_dicts[(filename,w,h)] = []
    for sem,ins in uni_labels:
        if ins<0: continue
        valid = np.logical_and(labels[:,0]==sem,
                               labels[:,1]==ins)
        if not len(valid): continue
        arg = args[valid].reshape(-1,2)
        
        x1, y1 = np.min(arg[:,0],axis=0), np.min(arg[:,1],axis=0)
        x2, y2 = np.max(arg[:,0],axis=0), np.max(arg[:,1],axis=0)
        det_dicts[(filename,w,h)].append([x1,y1,x2,y2,sem])
    return det_dicts

def gen_coco(dicts):
    categories_filed = []
   
    for i,cat in enumerate(LABELS):
        categories_filed.append({
                'id': i,
                'name': cat,
                'supercategory': 'yangtu'
                })
    main_dict = {
            'images': [],
            'annotations': [],
            'categories':categories_filed,
            }
    image_id, ann_id = 0, 0
    for (filename,w,h),boxes in dicts.items():
        main_dict['images'].append({
                'id': image_id,
                'file_name': filename,
                'width': w,
                'height': h
            })
        for box in boxes:
            x1,y1,x2,y2,label = box
            area = (x2 - x1) * (y2 - y1)
            main_dict['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': int(label),
                'segmentation': [],
                'area': area,
                'bbox': [x1,y1,x2-x1,y2-y1],
                'iscrowd': 0
            })
            ann_id += 1
        image_id += 1
    json.dump(main_dict, open("./val.cocojson", 'w'), indent=4)
        
    
    
if __name__ == "__main__":
    
    import os.path as osp
    import glob
    data_root = "dataset/svg"
    files = glob.glob(osp.join(data_root,"val", "*.svg"))
    inputs = []
    for svg_file in files:
        json_file = svg_file.replace('.svg','.json')
        if not osp.exists(svg_file) or not osp.exists(json_file): continue
        inputs.append([svg_file, json_file])
    import mmcv
    gt_dicts = mmcv.track_parallel_progress(process_gt, inputs, 16)
    gts = {}
    for det in gt_dicts: 
        gts.update(det)
    gen_coco(gts)
    
