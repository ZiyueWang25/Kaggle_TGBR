import util

class Base:
    lr0 = 1e-3  # initial learning rate (SGD=1E-2, Adam=1E-3)
    lrf = 0.2  # final OneCycleLR learning rate (lr0 * lrf), # 0.2
    momentum = 0.937  # SGD momentum/Adam beta1
    weight_decay = 0.0005  # optimizer weight decay 5e-4
    warmup_epochs = 3  # warmup epochs (fractions ok)
    warmup_momentum = 0.8  # warmup initial momentum
    warmup_bias_lr = 0.1  # warmup initial bias lr 
    box = 0.05  # box loss gain ?
    cls = 0.5  # cls loss gain ?
    cls_pw = 1.0  # cls BCELoss positive_weight 
    obj = 1.0  # obj loss gain (scale with pixels) ?
    obj_pw = 1.0  # obj BCELoss positive_weight 
    iou_t = 0.20  # IoU training threshold ?
    anchor_t = 4.0  # anchor-multiple threshold
    # anchors = 3  # anchors per output layer (0 to ignore)
    fl_gamma = 0.0  # focal loss gamma (efficientDet default gamma=1.5)
    hsv_h = 0 # 0.015  # image HSV-Hue augmentation (fraction) 0.015
    hsv_s = 0 # 0.7  # image HSV-Saturation augmentation (fraction) 0.7
    hsv_v = 0 # 0.4  # image HSV-Value augmentation (fraction) 0.4
    degrees = 0 # 10  # image rotation (+/- deg)  # 0
    translate = 0  # image translation (+/- fraction)
    scale = 0.0  # image scale (+/- gain) # 0.5
    shear = 0.0  # image shear (+/- deg) # 0.0
    perspective = 0  # image perspective (+/- fraction), range 0-0.001
    flipud = 0.5  # image flip up-down (probability)
    fliplr = 0.5  # image flip left-right (probability)
    mosaic = 1  # image mosaic (probability) # 1.0
    mixup = 0.2 # image mixup (probability) # 0.0
    copy_paste = 0.0  # segment copy-paste (probability)

class YOLOV5(Base):
    hsv_h = 0.015  # image HSV-Hue augmentation (fraction)
    hsv_s = 0.7  # image HSV-Saturation augmentation (fraction)
    hsv_v = 0.4  # image HSV-Value augmentation (fraction)
    degrees: 0.0  # image rotation (+/- deg)
    translate = 0.1  # image translation (+/- fraction)
    scale = 0.5  # image scale (+/- gain)
    shear = 0.0  # image shear (+/- deg)
    perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
    flipud = 0.5  # image flip up-down (probability)
    fliplr = 0.5  # image flip left-right (probability)
    mosaic = 1.0  # image mosaic (probability)
    mixup = 0.2  # image mixup (probability)
    copy_paste = 0.0  # segment copy-paste (probability)
    
class YOLOV5_LR5e3(YOLOV5):
    lr0 = 5e-3

class YOLOV5_LR5e4(YOLOV5):
    lr0 = 5e-4

class YOLOV5_SA_LR5e4(YOLOV5):
    # stronger augmentation
    mixup = 0.5
    lr0 = 5e-4
    
    
class sheep(YOLOV5):
    lr0 = 0.01
    lrf = 0.1
    mixup = 0.5
    
class YOLOV5_B4(Base):
    lr0 = 6e-4

class YOLOV5_B4_MU8(YOLOV5_B4):
    mixup = 0.8
    
def read_hyp_param(name):
    assert name in globals(), "name is not in " + str(globals())
    hyp_param = globals()[name]
    hyp_param = util.class2dict(hyp_param)
    return hyp_param