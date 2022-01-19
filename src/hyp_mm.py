import util

class FasterRCNN:
    model_type = "faster_rcnn"
    base_file = "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
    load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

class FasterRCNN_Pretrain(FasterRCNN):
    load_from = 'checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth' # pretrained

    
def read_hyp_param(name):
    assert name in globals(), "name is not in " + str(globals())
    hyp_param = globals()[name]
    hyp_param = util.class2dict(hyp_param)
    hyp_param['base_file'] = "mmdetection/configs/" + hyp_param["base_file"]
    hyp_param['load_from'] = "mmdetection/" +  hyp_param["load_from"]
    return hyp_param