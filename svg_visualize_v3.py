import json,os,glob
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from svgpathtools import parse_path
import re, math
from svgnet.data.svg import SVG_CATEGORIES

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
    #prettyxml = BeautifulSoup(ET.tostring(root, 'utf-8'), "xml").prettify()
    #with open(svg_path, "w") as f:
    #    f.write(prettyxml)

def visualSVG(parsing_list,labels,out_path,cvt_color=False):
    
    
    ind = 0
    for line in parsing_list:
        tag = line["tag"].split("svg}")[-1]
        assert tag in ['svg', 'g', 'path', 'circle', 'ellipse', 'text'],tag+" is error!!"
        if tag in ["path","circle",]:
            label = int(line["semanticIds"]) if "semanticIds" in line.keys() else -1
            label = labels[ind]
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
        

def process_dt(input):
    parsing_list,labels,out_path = input
    
    visualSVG(parsing_list,labels,out_path)
    svg2png(out_path)

def svg2png(svg_path, background_color="white", scale=7.0):
    '''
    Convert svg to png
    '''
    png_path = svg_path.replace(".svg",".png")
    # cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="white")
    command = "cairosvg {} -o {} -b {} -s {}".format(svg_path, png_path, background_color, scale)
    os.system(command)



def get_path(svg_lists):
    args, widths, gids, lengths, types = [], [], [], [], []
    COMMANDS = ['Line', 'Arc','circle', 'ellipse']
    for line in svg_lists:
       
        if "d" in line.keys():
            widths.append(line["stroke-width"])
            gid = int(line["gid"]) if "gid" in line.keys() else -1
            gids.append(gid)
            path_repre = parse_path(line['d'])
            inds = [0, 1/3, 2/3, 1.0]
            arg = []
            for ind in inds:
                point = path_repre.point(ind)
                arg.extend([point.real,point.imag])
            args.append(arg)
            length = path_repre.length()
            lengths.append(length)
            path_type = path_repre[0].__class__.__name__
            types.append(COMMANDS.index(path_type))
        elif "r" in line.keys():
            widths.append(line["stroke-width"])
            gid = int(line["gid"]) if "gid" in line.keys() else -1
            gids.append(gid)
            cx = float(line['cx'])
            cy = float(line['cy'])
            r = float(line['r'])
            arg = []
            thetas = [0,math.pi/2, math.pi, 3 * math.pi/2,]
            for theta in thetas:
                x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
                arg.extend([x,y])
            args.append(arg)
            circle_len = 2 * math.pi * r
            lengths.append(circle_len)
            types.append(COMMANDS.index("circle"))
    return widths, gids, args, lengths,types
            




    


if __name__ == "__main__":
    from svgnet.evaluation import InstanceEval
    from svgnet.util  import get_root_logger
    instance_eval = InstanceEval(num_classes=35,
                                 ignore_label=35,gpu_num=1)
    logger = get_root_logger()
    
    res_file = "./sem_ins_split_val.npy"
    save_dir = "./spv2-norgb-val"
    os.makedirs(save_dir,exist_ok=True)
    
    detections = np.load(res_file,allow_pickle=True)
    import tqdm
    inputs = []
    coco_res = []
    for det in tqdm.tqdm(detections):
        
        svg_path = det["filepath"].replace("_s2.svg", ".svg")
        
        #if "241f" not in svg_path: continue
        assert os.path.exists(svg_path) is True,"svg_file not exists!!!"
        parsing_list = svg_reader(svg_path)
        widths, gids, args, lengths, types = get_path(parsing_list)
        widths, gids, lengths, types = np.array(widths), np.array(gids), np.array(lengths), np.array(types)
        coords = np.array(args).reshape(-1, 4,2)
        det["instances"] = []
        ins_outs = det["ins"]
        if not len(ins_outs): continue
        shape = ins_outs[0]["masks"].shape[0]
        sem_out = np.full_like(np.zeros(shape),-1)
        
        for instance in ins_outs:
            masks, labels = instance["masks"],instance["labels"]
            scores = instance["scores"]
            if scores<0.1: continue
            sem_out[masks] = labels
            det["instances"].append({"masks":masks, "labels":labels,"scores":scores}) 
        
        
        coco_res.append({'filepath': det['filepath'],'instances': det['instances']})
        instance_eval.update(det["instances"],det["targets"],det["lengths"])    
        
        out_path = os.path.join(save_dir,svg_path.split("/")[-1])
        inputs.append([parsing_list,sem_out.astype(np.int64),out_path])
    instance_eval.get_eval(logger)
    np.save("coco_res_val.npy", coco_res)
    import mmcv
    mmcv.track_parallel_progress(process_dt,inputs,16)
    
    

    
    
    
    
    
    
