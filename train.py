import os
from pathlib import Path
import argparse
import logging
import time
import ast 
import yaml
import sys
import json
import shutil
import subprocess
sys.path.append("./src")
sys.path.append("./notebook/yolov5")

import mmcv
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import inference_detector

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import util
import hyp
import hyp_mm


class Pre:
    def __init__(self, params):
        self.params = params        
    
    def prepare_data(self):
        logging.debug("prepare_data")
        self.df = pd.read_csv(self.params['root_dir'] / 'train.csv')
        if self.params["debug"]:
            np.random.seed(2020)
            num_sample = self.df.shape[0] // 10
            self.df = self.df.sample(num_sample)
        self.df = self.df.apply(lambda x: util.get_path(x, self.params), axis=1)
        self.df['annotations'] = self.df['annotations'].apply(lambda x: ast.literal_eval(x))
        self.df['has_annotations'] = self.df['annotations'].apply(len) > 0        
        self.df['num_bbox'] = self.df['annotations'].apply(len)        
        data = (self.df.num_bbox>0).value_counts(normalize=True) * 100
        logging.debug(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")
        self.cv_split()
        util.seed_torch(self.params["seed"])
        if self.params['remove_nobbox']:
            logging.info("Remove no bbox instance")
            fold = self.params["fold"]
            # keep the background in validation fold
            self.df = self.df.query(f"num_bbox>0 or fold == {fold}")
        elif self.params['reduce_nobbox'] > 0:
            reduce_nobbox = self.params['reduce_nobbox']
            logging.info(f"reduce_nobbox {reduce_nobbox:.2%}, current size: {self.df.shape[0]}")
            fold = self.params["fold"]
            IS_nobbox_index = self.df.query(f"num_bbox==0 or fold != {fold}").index
            drop_size = int(len(IS_nobbox_index) * reduce_nobbox)
            drop_index = np.random.choice(IS_nobbox_index, size=drop_size, replace=False)
            self.df.drop(drop_index, axis=0, inplace=True)
            self.df.reset_index(inplace=True, drop=True)
            logging.info(f"after reducing, df size: {self.df.shape[0]}")
            
        self.df['bboxes'] = self.df.annotations.apply(util.get_bbox)
        self.df['width']  = 1280
        self.df['height'] = 720    
        util.seed_torch(self.params["seed"])
        self.create_dataset()
        if self.params["copy_image"]:
            self.copy_image()
            self.create_yolo_format()
        
        return 
    
    def cv_split(self):
        logging.debug("cv_split")   
        split_cri = self.params["cv_split"]
        if split_cri == "subsequence":     
            diff_place = (self.df["has_annotations"] + self.df["sequence"]).diff()
            diff_place = diff_place.shift(-1)
            diff_place.iloc[-1] = 1
            diff_place_filter = diff_place[diff_place!=0] 
            diff_place_filter[:] =1
            subsequence_id_place = diff_place_filter.cumsum()
            self.df["subsequence_id"] = np.nan
            self.df.loc[subsequence_id_place.index, "subsequence_id"] = subsequence_id_place.values
            self.df["subsequence_id"] = self.df["subsequence_id"].fillna(method="backfill")        
                    
            skf = StratifiedGroupKFold(n_splits=5)
            self.df = self.df.reset_index(drop=True)
            self.df['fold'] = -1
            for fold, (_, val_idx) in enumerate(skf.split(self.df, groups=self.df['subsequence_id'], y=self.df["has_annotations"])):
                self.df.loc[val_idx, 'fold'] = fold
        elif split_cri == "video_id":
            self.df["fold"] = self.df["video_id"]
        elif split_cri == "sequence":
            skf = StratifiedGroupKFold(n_splits=5)
            self.df = self.df.reset_index(drop=True)
            self.df['fold'] = -1
            for fold, (_, val_idx) in enumerate(skf.split(self.df, groups=self.df['sequence'], y=self.df["has_annotations"])):
                self.df.loc[val_idx, 'fold'] = fold
        else:
            logging.error(f"{split_cri} not in subsequence, video_id, sequence")
            
        logging.debug(self.df.fold.value_counts())
        return        
    
    def create_dataset(self):
        logging.debug("create_dataset")
        train_files = []
        val_files   = []
        fold = self.params["fold"]
        self.train_df = self.df.query(f"fold!={fold}")
        self.valid_df = self.df.query(f"fold=={fold}")
        train_files += list(self.train_df.image_path.unique())
        val_files += list(self.valid_df.image_path.unique())
        logging.debug(f"""" len(train_files): {len(train_files)}, len(val_files): {len(val_files)}""")
        self.add_config()

    
    def add_config(self):
        if self.params["tools"] == "yolov5":
            logging.debug("add_config")
            with open(self.params["cfg_dir"] / 'train.txt', 'w') as f:
                for path in self.train_df.image_path.tolist():
                    f.write(str(path.resolve())+'\n')
            with open(self.params["cfg_dir"] / 'val.txt', 'w') as f:
                for path in self.valid_df.image_path.tolist():
                    f.write(str(path.resolve())+'\n')

            data = dict(
                path = str(self.params["cfg_dir"].resolve()),
                train = str((self.params["cfg_dir"] / 'train.txt').resolve()),
                val = str((self.params["cfg_dir"] / 'val.txt').resolve()),
                nc = 1,
                names = ['cots'],
                )

            with open(self.params["cfg_dir"] / 'bgr.yaml', 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False)

            f = open(self.params["cfg_dir"] / 'bgr.yaml', 'r')
            logging.debug('\nyaml:')
            logging.debug(f.read())        
        else:
            json_train = util.coco(self.train_df)
            json_valid = util.coco(self.valid_df)
            with open(self.params["cfg_dir"] / 'annotations_train.json', 'w', encoding='utf-8') as f:
                json.dump(json_train, f, ensure_ascii=True, indent=4)
                
            with open(self.params["cfg_dir"] / 'annotations_valid.json', 'w', encoding='utf-8') as f:
                json.dump(json_valid, f, ensure_ascii=True, indent=4)  
                                  
    
    def create_yolo_format(self):
        logging.debug("create_yolo_format")
        cnt = 0
        all_bboxes = []
        for row_idx in range(self.df.shape[0]):
            row = self.df.iloc[row_idx]
            image_height = row.height
            image_width  = row.width
            bboxes_coco  = np.array(row.bboxes).astype(np.float32).copy()
            num_bbox     = len(bboxes_coco)
            labels       = [0]*num_bbox
            ## Create Annotation(YOLO)
            with open(row.label_path, 'w') as f:
                if num_bbox<1:
                    annot = ''
                    f.write(annot)
                    cnt+=1
                    continue
                bboxes_yolo  = util.coco2yolo(image_height, image_width, bboxes_coco)
                bboxes_yolo  = np.clip(bboxes_yolo, 0, 1)
                all_bboxes.extend(bboxes_yolo)
                for bbox_idx in range(len(bboxes_yolo)):
                    annot = [str(labels[bbox_idx])]+ list(bboxes_yolo[bbox_idx].astype(str))+(['\n'] if num_bbox!=(bbox_idx+1) else [''])
                    annot = ' '.join(annot)
                    annot = annot.strip(' ')
                    f.write(annot)
        logging.debug(f'Missing: {cnt}') 
        return       
    
    def copy_image(self):
        logging.debug("copy_image")
        from joblib import Parallel, delayed
        image_paths = self.df.old_image_path.tolist()
        _ = Parallel(n_jobs=-1, backend='threading')(delayed(util.make_copy)(path, self.params) for path in image_paths)        
        return
    
def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', type=str, default='TGBR')
    parser.add_argument('--tools', type=str, default='yolov5') # or mmdetection
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--fold', type=int, nargs='+', default=0)        
    parser.add_argument('--data_path', type=str, default="../data/tensorflow-great-barrier-reef/")
    parser.add_argument('--remove_nobbox', action='store_true')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--copy_image', action='store_true')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--cv_split', type=str, default="subsequence") # subsequence, video_id, sequence
    parser.add_argument('--reduce_nobbox', type=float, default=0)
    

    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weights', type=str, default="yolov5s.pt")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default="AdamW")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')

    # yolov5
    parser.add_argument("--hyp_name", type=str, default="Base")
    
    # for inference
    parser.add_argument("--img_size", type=int, default=1280) # to save exp parameters
    parser.add_argument("--no_train", action="store_true") # to save exp parameters
    
    # upload to kaggle
    parser.add_argument("--upload", action='store_true')   

    args = parser.parse_args()
    # convert to dictionary
    params = vars(args)
    prefix = time.strftime("%m%d") + "_"
    params["exp_name"] = prefix + params["exp_name"]
    params['data_path'] = Path(params['data_path']).resolve()
    params['root_dir'] = params['data_path']
    params['image_dir'] = params['data_path'] / 'images'    
    params['label_dir'] = params['data_path'] / 'labels'    
    
    if not os.path.exists(params['label_dir']):
        os.makedirs(params['label_dir'])
    if not os.path.exists(params['image_dir']):
        os.makedirs(params['image_dir'])
        
    output_dir = Path(os.path.abspath("./output/"))
    
    output_dir = output_dir / params["exp_name"]
    params['ckpt_path'] = output_dir / "TGBR" / params["exp_name"] /  "weights" / "best.pt"
    
    cfg_dir = output_dir / "config"
    log_file = output_dir / "log.txt"
    params['output_dir'] = output_dir
    params['cfg_dir'] = cfg_dir
    params["log_file"] = log_file
    if params["hyp_name"] != "None":
        if params["tools"] == "yolov5":
            params["hyp_param"] = hyp.read_hyp_param(params["hyp_name"])
            params["hyp_file"] = cfg_dir / "hyp.yaml"
        elif params["tools"] == "mmdetection":
            params["hyp_param"] = hyp_mm.read_hyp_param(params["hyp_name"])
            params["hyp_file"] = cfg_dir / "hyp.yaml"
    else:
        params["hyp_file"] = ""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)            
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    if params["hyp_name"] != "None":
        util.write_hyp(params)
    util.save_yaml(params, cfg_dir / "params.yaml")
    return params

def call_subprocess(params):
    if params["tools"] == "yolov5":
        script_path = Path("./yolov5/train.py").resolve()
        data_path = (params["cfg_dir"] / "bgr.yaml").resolve()
        input_dir = Path("./input/").resolve()
        hyp_file = params["hyp_file"].resolve() if params["hyp_file"] != "" else ""
        logging.debug(str(script_path))
        logging.debug(str(data_path))
        logging.debug(str(input_dir))
        os.chdir(params['output_dir'])
        args = ["python3", str(script_path), 
                        "--img", str(params['img_size']),
                        "--batch", str(params['batch']),
                        "--data", str(data_path),
                        "--epochs", str(params['epochs']),
                        "--weights", str(input_dir / params['weights']),
                        "--workers", str(params['workers']),
                        "--patience", str(params['patience']),
                        "--optimizer", str(params['optimizer']),
                        "--name", params["exp_name"],
                        "--project", params["project"],
                        '--device', params['device'],
                        ]
        if hyp_file != "":
            args.extend(["--hyp", str(hyp_file)])
        if params["sync_bn"]:
            args.extend(["--sync-bn"])
        subprocess.call(args)    
    elif params['tools'] == 'mmdetection':
        print(params['hyp_param'])
        cfg = util.mmcfg_from_param(params)
        meta = dict()
        meta['config'] = cfg.pretty_text
        datasets = [build_dataset(cfg.data.train)] #, build_dataset(cfg.data.val)] # no valid works ok
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        model.CLASSES = datasets[0].CLASSES
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))                        
        train_detector(model, datasets, cfg, distributed=False, validate=True, meta = meta)   
             

def main():
    params = parse_args()
    util.create_logger(params['log_file'], filemode='a')
    print("\nLOGGING TO: ", params['log_file'], "\n")
    if not params["no_train"]:
        torch.backends.cudnn.benchmark = True
        pre = Pre(params)
        pre.prepare_data()
        call_subprocess(params)
    if params["upload"]:
        util.upload(params)

if __name__ == "__main__":
    main()
