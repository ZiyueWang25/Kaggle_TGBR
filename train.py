import os
from pathlib import Path
import argparse
import logging
import time
import ast 
import yaml
import sys
import subprocess
sys.path.append("./src")
sys.path.append("./notebook/yolov5")

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import util
import hyp


class Pre:
    def __init__(self, params):
        self.params = params        
    
    def prepare_data(self):
        logging.debug("prepare_data")
        self.df = pd.read_csv(self.params['root_dir'] / 'train.csv')
        self.df = self.df.apply(lambda x: util.get_path(x, self.params), axis=1)
        self.df['annotations'] = self.df['annotations'].apply(lambda x: ast.literal_eval(x))
        self.df['num_bbox'] = self.df['annotations'].apply(len)        
        data = (self.df.num_bbox>0).value_counts(normalize=True) * 100
        logging.debug(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")
        if self.params['remove_nobbox']:
            logging.info("Remove no bbox instance")
            self.df = self.df.query("num_bbox>0")
        self.df['bboxes'] = self.df.annotations.apply(util.get_bbox)
        self.df['width']  = 1280
        self.df['height'] = 720    
        self.cv_split()
        util.seed_torch(self.params["seed"])
        self.create_dataset()
        return 
    
    def cv_split(self):
        logging.debug("cv_split")
        kf = GroupKFold(n_splits = 5)
        self.df = self.df.reset_index(drop=True)
        self.df['fold'] = -1
        for fold, (_, val_idx) in enumerate(kf.split(self.df, y = self.df.video_id.tolist(), groups=self.df.sequence)):
            self.df.loc[val_idx, 'fold'] = fold
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
        logging.debug('Missing:',cnt) 
        return       
    
    def copy_image(self):
        logging.debug("copy_image")
        from joblib import Parallel, delayed
        image_paths = self.df.old_image_path.tolist()
        _ = Parallel(n_jobs=-1, backend='threading')(delayed(util.make_copy)(path, self.params) for path in image_paths)        
        return
    
def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', type=str, default='yolov5s')
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--fold', type=int, nargs='+', default=0)        
    parser.add_argument('--data_path', type=str, default="../data/tensorflow-great-barrier-reef/")
    parser.add_argument('--remove_nobbox', action='store_true')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument("--hyp_name", type=str, default="Base")
    parser.add_argument('--copy_image', action='store_true')
    
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--weights', type=str, default="yolov5s.pt")
    parser.add_argument('--workers', type=int, default=8)
    
    # for inference
    parser.add_argument("--no_train", action="store_true") # to save exp parameters
    parser.add_argument('--repo', type=str, default='/kaggle/input/yolov5-lib-ds')
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--img_size", type=int, default=1280)

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
    params['ckpt_path'] = output_dir / params["weights"].split(".")[0] / params["exp_name"] /  "weights" / "best.pt"
    
    cfg_dir = output_dir / "config"
    log_file = output_dir / "log.txt"
    params['output_dir'] = output_dir
    params['cfg_dir'] = cfg_dir
    params["log_file"] = log_file
    params["hyp_param"] = hyp.read_hyp_param(params["hyp_name"])
    params["hyp_file"] = cfg_dir / "hyp.yaml"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)            
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    util.write_hyp(params)
    util.save_pickle(params, cfg_dir / "params.pkl")
    return params

def call_subprocess(params):
    script_path = Path("./notebook/yolov5/train.py").resolve()
    data_path = (params["cfg_dir"] / "bgr.yaml").resolve()
    input_dir = Path("./input/").resolve()
    hyp_file = params["hyp_file"].resolve()
    logging.debug(str(script_path))
    logging.debug(str(data_path))
    logging.debug(str(input_dir))
    os.chdir(params['output_dir'])
    subprocess.call(["python3", str(script_path), 
                     "--img", str(params['img_size']),
                     "--batch", str(params['batch']),
                     "--data", str(data_path),
                     "--hyp", str(hyp_file),
                     "--epochs", str(params['epochs']),
                     "--weights", str(input_dir / params['weights']),
                     "--workers", str(params['workers']),
                     "--name", params["exp_name"],
                     "--project", params["project"]
                     ]
                    )    

def main():
    params = parse_args()
    util.create_logger(params['log_file'], filemode='a')
    print("\nLOGGING TO: ", params['log_file'], "\n")
    if not params["no_train"]:
        pre = Pre(params)
        pre.prepare_data()
        if params["copy_image"]:
            pre.copy_image()
        call_subprocess(params)
    
if __name__ == "__main__":
    main()
