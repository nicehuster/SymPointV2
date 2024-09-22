
import glob, os, json
import os.path as osp
import xml.etree.ElementTree as ET
import mmcv
import numpy as np
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

root = "dataset/svg/test"
save_dir = "./test_gt_vis"
os.makedirs(save_dir, exist_ok=True)
def collect_dataset():
    
    datasets = []
    svg_file = glob.glob(
        osp.join(root, "*.svg")
    )
    for svg_file in svg_file:
        if "_res" in svg_file: continue
        if "_s2" in svg_file: continue
        json_file = svg_file.replace(".svg", '_s2.json')
        if not osp.exists(json_file): continue
        #if "0006-0015" not in json_file: continue
        datasets.append({
            "json": json_file,
            "svg": svg_file
        })
    
    print("datasets: ", len(datasets))
    return datasets

def svg_reader(svg_path):
    svg_list = list()
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        print("Read{} failed!".format(svg_path))
        return svg_list
    root = tree.getroot()
    for elem in root.iter():
        line = elem.attrib
        line['tag'] = elem.tag
        svg_list.append(line)
    return svg_list

def svg_writer(svg_list, svg_path):
    for idx, line in enumerate(svg_list):
        tag = line["tag"]
        line.pop("tag")
        if idx == 0:
            root = ET.Element(tag)
            root.attrib = line
        else:
            if "}g" in tag:
                group = ET.SubElement(root, tag)
                group.attrib = line
            else:
                node = ET.SubElement(group, tag)
                node.attrib = line
     
    from xml.dom import minidom
    reparsed = minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="\t")
    f = open(svg_path,'w',encoding='utf-8')
    f.write(reparsed)
    f.close()           


def visualSVG(parsing_list,out_path,cvt_color=False):
    
    
    ind = 0
    for line in parsing_list:
        tag = line["tag"].split("svg}")[-1]
        assert tag in ['svg', 'g', 'path', 'circle', 'ellipse', 'text'],tag+" is error!!"
        if tag in ["path","circle",]:
            label = int(line["semanticId"])-1 if "semanticId" in line.keys() else -1
            color = SVG_CATEGORIES[label]["color"]
            line["stroke"] = "rgb({:d},{:d},{:d})".format(color[0],color[1],color[2]) 
            line["fill"] = "none"
            line["stroke-width"] = "0.2"
            ind += 1
     
        if tag == "svg":
            viewBox = line["viewBox"]
            viewBox = viewBox.split(" ")
            line["viewBox"] = " ".join(viewBox)
            if cvt_color:
                line["style"] = "background-color: #255255255;"

    svg_writer(parsing_list, out_path)
    return out_path
        
def svg2png(svg_path, background_color="white", scale=7.0):
    '''
    Convert svg to png
    '''
    png_path = svg_path.replace(".svg",".png")
    # cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="white")
    command = "cairosvg {} -o {} -b {} -s {}".format(svg_path, png_path, background_color, scale)
    os.system(command)

def visualize(data):
    
    svg_file, json_file = data["svg"], data["json"]
    parsing_list = svg_reader(svg_file)
    out_path = svg_file.replace(".svg", "_res.svg")
    visualSVG(parsing_list, out_path)
    svg2png(out_path)
    img_file = out_path.replace(".svg",".png")
    image = Image.open(img_file)
    draw = ImageDraw.Draw(image,'RGBA')
    data = json.load(open(json_file))
    coords = np.array(data["args"]).reshape(-1,8)
    seg = np.array(data["semanticIds"])
    ins = np.array(data["instanceIds"])
    labels = np.concatenate([seg[:,None],ins[:,None]],axis=1)
    
    uni_labels = np.unique(labels, axis=0)
    for ulabel in uni_labels:
        sem, ins = ulabel
        if sem>=30: continue
        mask = np.logical_and(labels[:, 0]==sem,
                              labels[:, 1]==ins)
        arg = coords[mask].reshape(-1, 2)
        x1, y1 = np.min(arg[:,0],axis=0), np.min(arg[:,1],axis=0)
        x2, y2 = np.max(arg[:,0],axis=0), np.max(arg[:,1],axis=0)
        #vis
        color = SVG_CATEGORIES[int(sem)]["color"]
        draw.rectangle([x1*7,y1*7,x2*7,y2*7],fill=tuple(color+[32]),width=2)
        text = '{}'.format(str(sem))
        draw.text((x1*7,y1*7),text,align='right',fill=(0,0,0))
    
    filename = os.path.basename(img_file).replace(".png","_res.png")
    save = os.path.join(save_dir, filename)
    image.save(save)  

datasets = collect_dataset()
#for data in datasets: visualize(data)
mmcv.track_parallel_progress(visualize, datasets, 16)
     
    
    
    
    