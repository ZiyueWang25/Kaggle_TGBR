import ast 
import sys
sys.path.append("./src")
sys.path.append("./notebook/yolov5")

import pandas as pd
import numpy as np
import util

import cv2
import tqdm

def run(params):
    print("Read Data")
    df = pd.read_csv(params["root_dir"] / 'train.csv')
    df = df.apply(lambda x: util.get_path(x, params), axis=1)
    df['annotations'] = df['annotations'].apply(lambda x: ast.literal_eval(x))
    df['num_bbox'] = df['annotations'].apply(lambda x: len(x))
    util.seed_torch(params["seed"])    
    colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for idx in range(1)]    
    model = util.load_model(params)
    print("predict training data")
    image_paths = df[df.num_bbox>1].sample(100).image_path.tolist()    
    for idx, path in enumerate(image_paths):
        img = cv2.imread(str(path))[...,::-1]
        bboxes, confis = util.predict(model, img, size=params["img_size"], augment=params["augment"])
        display(util.show_img(img, bboxes, colors, bbox_format='coco'))
        if idx>5:
            break
    
    if params["run_test"]:
        print("Run test")
        import greatbarrierreef
        env = greatbarrierreef.make_env()# initialize the environment
        iter_test = env.iter_test()      # an iterator which loops over the test set and sample submission
        model = util.load_model(params)
        for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
            bboxes, confs = util.predict(model, img, size=params["img_size"], augment=params["augment"])
            annot = util.format_prediction(bboxes, confs)
            pred_df['annotations'] = annot
            env.predict(pred_df)
            if idx<3:
                display(util.show_img(img, bboxes, colors, bbox_format='coco'))