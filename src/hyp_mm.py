import util
class Base:
    # https://www.codetd.com/en/article/12791298
    script_f = "mmdetection"
    loss_fnc = "L1Loss" # L1Loss, GIoULoss
    nms="nms" # nms, soft_nsm
    sampler = 'RandomSampler' # RandomSampler, OHEMSampler, 
    
class FasterRCNN(Base):
    model_type = "faster_rcnn"
    base_file = "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
    load_from = 'faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth' # pretrained
    
class CRCNN(Base):
    model_type = "cascade_rcnn"
    base_file = "cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py"
    load_from = 'cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
    
    
        
class swin(Base):
    script_f = "mmdetection"# "Swin-Transformer-Object-Detection"
    model_type = "swin"
    #base_file = "swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
    #load_from = "cascade_mask_rcnn_swin_small_patch4_window7.pth"
    base_file = 'swin/TFGBR_swin_base_faster_rcnn_fp16.py'
    load_from = "swin_small_patch4_window7_224.pth"
    lr = 0.0001

class vfnet(Base):
    model_type = 'vfnet'
    base_file = 'vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
    load_from = "vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth"
    
# class swin2(Base):
#     model_type = "swin"
#     base_file = 'swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'    
#     load_from = 'cascade_mask_rcnn_swin_small_patch4_window7.pth'
#     lr = 0.0001
    
def read_hyp_param(name):
    assert name in globals(), "name is not in " + str(globals())
    hyp_param = globals()[name]
    hyp_param = util.class2dict(hyp_param)
    script_f = hyp_param['script_f']
    hyp_param['base_file'] = script_f + "/configs/" + hyp_param["base_file"]
    hyp_param['load_from'] = script_f + "/checkpoints/" +  hyp_param["load_from"]
    return hyp_param