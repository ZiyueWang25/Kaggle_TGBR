# python3 train.py --exp_name yolov5s_fold0 --fold 0 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10 --copy_image
# python3 train.py --exp_name yolov5s_fold1 --fold 1 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10
# python3 train.py --exp_name yolov5s_fold2 --fold 2 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10
# python3 train.py --exp_name yolov5s_fold3 --fold 3 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10
# python3 train.py --exp_name yolov5s_fold4 --fold 4 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10
# python3 train.py --exp_name yolov5x_fold0 --fold 0 --remove_nobbox --batch 16 --epochs 20 --weights yolov5x.pt --workers 10 --upload # batch size too big
# python3 train.py --exp_name yolov5l_fold0 --fold 0 --remove_nobbox --batch 16 --epochs 20 --weights yolov5l.pt --workers 10
# python3 train.py --exp_name yolov5m_fold0 --fold 0 --remove_nobbox --batch 16 --epochs 20 --weights yolov5m.pt --workers 10
# python3 train.py --exp_name yolov5x_fold0 --fold 0 --remove_nobbox --batch 8 --epochs 20 --weights yolov5x.pt --workers 10
# python3 train.py --exp_name yolov5l_fold0 --fold 0 --remove_nobbox --batch 8 --epochs 20 --weights yolov5l.pt --workers 10 # good to train
# python3 train.py --exp_name yolov5s_fold0_default_300ep --fold 0 --remove_nobbox --batch 16 --epochs 300 --hyp_name None --weights yolov5s.pt --workers 10

# python3 train.py --exp_name yolov5s_fold0_default_300ep_newCV_allbox --fold 0 --batch 16 --epochs 300 --hyp_name None --weights yolov5s.pt --workers 10 --copy_image

# python3 train.py --exp_name yolov5s_fold0_default_300ep_newcv_nobbox --fold 0 --batch 16 --epochs 50 --hyp_name None --weights yolov5s.pt --workers 12 --copy_image --remove_nobbox

# python3 train.py --exp_name yolov5l_fold0_default_50ep_newcv_nobbox --fold 0 --batch 8 --epochs 50 --hyp_name None --weights yolov5l.pt --workers 12 --copy_image --remove_nobbox


# python3 train.py --exp_name yolov5s_fold1_default_100ep_newcv_nobbox --fold 1 --batch 16 --epochs 100 --hyp_name None --weights yolov5s.pt --workers 12 --copy_image --remove_nobbox
# python3 train.py --exp_name yolov5s_fold2_default_100ep_newcv_nobbox --fold 2 --batch 16 --epochs 100 --hyp_name None --weights yolov5s.pt --workers 12 --copy_image --remove_nobbox
# python3 train.py --exp_name yolov5s_fold3_default_100ep_newcv_nobbox --fold 3 --batch 16 --epochs 100 --hyp_name None --weights yolov5s.pt --workers 12 --copy_image --remove_nobbox
# python3 train.py --exp_name yolov5s_fold4_default_100ep_newcv_nobbox --fold 4 --batch 16 --epochs 100 --hyp_name None --weights yolov5s.pt --workers 12 --copy_image --remove_nobbox


# python3 train.py --exp_name yolov5s_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name Base --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50

# python3 train.py --exp_name yolov5n_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name Base --weights yolov5n.pt --workers 12 --remove_nobbox --patience 50

# python3 train.py --exp_name yolov5s_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name Base --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50

# python3 train.py --exp_name yolov5s_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name LR0005 --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50

# python3 train.py --exp_name yolov5s_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name LRF001 --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50


# python3 train.py --exp_name yolov5l_fold0_Base --fold 0 --batch 8 --epochs 300 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox --patience 50
# python3 train.py --exp_name yolov5l_fold1_Base --fold 1 --batch 8 --epochs 300 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox --patience 50

# check error
# python3 train.py --exp_name yolov5n_fold0_Base --fold 0 --batch 16 --epochs 50 --hyp_name Base --weights yolov5n.pt --workers 12 --remove_nobbox --patience 50 --debug

# python3 train.py --exp_name yolov5l_fold0_Base_fix --fold 0 --batch 8 --epochs 100 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox --patience 50

# python3 train.py --exp_name yolov5s_fold0_Base_clahe --fold 0 --batch 16 --epochs 100 --hyp_name Base --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50

# python3 train.py --exp_name yolov5n_fold0_debug --fold 0 --batch 24 --hyp_name Base --weights yolov5n.pt --workers 12 --debug

# python3 train.py --exp_name yolov5s_fold0_new_hyp --fold 0 --batch 16 --hyp_name Base --weights yolov5s.pt --workers 12
# python3 train.py --exp_name yolov5l_fold0_new_hyp_remove_noaug --fold 0 --batch 8 --hyp_name NoAug --weights yolov5l.pt --workers 12 --remove_nobbox
# python3 train.py --exp_name yolov5l_fold0_new_hyp_remove --fold 0 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5l_fold0_nobbox_cvsequence --fold 0 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox --cv_split sequence

# python3 train.py --exp_name yolov5l_fold0_nobbox_cvvideo --fold 0 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox --cv_split video_id

# python3 train.py --exp_name yolov5s_fold0_new_hyp_remove_noaug --fold 0 --batch 16 --hyp_name NoAug --weights yolov5s.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5s_fold0_new_hyp_remove_mosaic --fold 0 --batch 16 --hyp_name Mosaic --weights yolov5s.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5s_fold0_new_hyp_remove_mosaicflip --fold 0 --batch 16 --hyp_name MosaicFlip --weights yolov5s.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5s_fold0_new_hyp_remove_mosaicflipmixup --fold 0 --batch 16 --hyp_name MosaicFlipMixUp --weights yolov5s.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5s_fold0_new_hyp_remove_hsv --fold 0 --batch 16 --hyp_name HSV --weights yolov5s.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5l_fold0_new_hyp_remove_1 --fold 0 --batch 8 --hyp_name MosaicFlipMixUp --weights yolov5l.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5l_fold0_new_hyp_remove_2 --fold 0 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5l_fold0_new_hyp_remove_1r2e3 --fold 0 --batch 8 --hyp_name Base_lr2e3 --weights yolov5l.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5l_fold0_new_hyp_remove_1r5e4 --fold 0 --batch 8 --hyp_name Base_lr5e4 --weights yolov5l.pt --workers 12 --remove_nobbox

# python3 train.py --exp_name yolov5l_fold0_newBase --fold 0 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox
# python3 train.py --exp_name yolov5n_fold0_newBase_MixUp0_check --fold 0 --batch 32 --hyp_name MixUp0 --weights yolov5n.pt --workers 12 --remove_nobbox
# python3 train.py --exp_name yolov5l_fold1_newBase --fold 1 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox
# python3 train.py --exp_name yolov5l_fold2_newBase --fold 2 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox
# python3 train.py --exp_name yolov5l_fold3_newBase --fold 3 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox
# python3 train.py --exp_name yolov5l_fold4_newBase --fold 4 --batch 8 --hyp_name Base --weights yolov5l.pt --workers 12 --remove_nobbox

#python3 train.py --exp_name yolov5l_fold0_4K --img_size 2560 --fold 0 --batch 4 --hyp_name YOLOV5 --weights yolov5l.pt --epochs 15 --optimizer SGD --workers 8 --cv_split video_id  --sync-bn --device 0,1




# python3 train.py --exp_name fastrcnn_pretrain --img_size 1280 --fold 0 --batch 4 --tools mmdetection --hyp_name FasterRCNN_Pretrain --epochs 15 --workers 8 --cv_split subsequence --remove_nobbox

#python3 train.py --exp_name yolov5l_fold0_new_hyp_remove_1_noClahe --fold 0 --batch 8 --hyp_name MosaicFlipMixUp --weights yolov5l.pt --workers 12 --remove_nobbox
#python3 train.py --exp_name yolov5l_fold0_new_hyp_remove_1_Clahe --fold 0 --batch 8 --hyp_name MosaicFlipMixUp --weights yolov5l.pt --workers 12 --remove_nobbox --use-clahe


# python3 train.py --exp_name fastrcnn --img_size 1280 --fold 0 --batch 4 --tools mmdetection --hyp_name FasterRCNN --epochs 15 --workers 8 --remove_nobbox

# python3 train.py --exp_name fastrcnn_pretrain --img_size 1280 --fold 0 --batch 10 --tools mmdetection --hyp_name FasterRCNN_Pretrain --epochs 15 --workers 10 --remove_nobbox

# python3 train.py --exp_name fastrcnn_pretrain --img_size 1280 --fold 1 --batch 10 --tools mmdetection --hyp_name FasterRCNN_Pretrain --epochs 15 --workers 10 --remove_nobbox

# python3 train.py --exp_name fastrcnn_pretrain --img_size 1280 --fold 2 --batch 10 --tools mmdetection --hyp_name FasterRCNN_Pretrain --epochs 15 --workers 10 --remove_nobbox

#python3 train.py --exp_name swin_debug --debug --img_size 1280 --fold 4 --batch 3 --tools mmdetection --hyp_name swin  --epochs 15 --workers 2 --remove_nobbox
#python3 train.py --exp_name swin --img_size 1280 --fold 4 --batch 2 --tools mmdetection --hyp_name swin  --epochs 15 --workers 4 --remove_nobbox
#python3 train.py --exp_name frcnn_newlr --fold 4 --batch 10 --tools mmdetection --hyp_name FasterRCNN  --epochs 30 --workers 4 --remove_nobbox
#python3 train.py --exp_name swin_newlr --fold 4 --batch 2 --tools mmdetection --hyp_name swin  --epochs 15 --workers 4 --remove_nobbox

# python3 train.py --exp_name frcnn_normallr --fold 4 --batch 10 --tools mmdetection --hyp_name FasterRCNN  --epochs 12 --workers 8 --remove_nobbox

#python3 train.py --exp_name frcnn_2 --fold 4 --batch 10 --tools mmdetection --hyp_name FasterRCNN  --epochs 12 --workers 8 --remove_nobbox
#python3 train.py --exp_name frcnn_giou --fold 4 --batch 10 --tools mmdetection --hyp_name FasterRCNN_GIoU  --epochs 12 --workers 8 --remove_nobbox
#python3 train.py --exp_name frcnn_giou_softnms --fold 4 --batch 10 --tools mmdetection --hyp_name FasterRCNN_GIoU_softNMS  --epochs 12 --workers 8 --remove_nobbox
#python3 train.py --exp_name frcnn_giou_softnms_ohem --fold 4 --batch 10 --tools mmdetection --hyp_name FasterRCNN_GIoU_softNMS_OHEM  --epochs 12 --workers 8 --remove_nobbox
#python3 train.py --exp_name frcnn_albu --fold 4 --batch 10 --tools mmdetection --hyp_name FasterRCNN  --epochs 12 --workers 8 --remove_nobbox
#python3 train.py --exp_name frcnn_albu5 --fold 4 --batch 6 --tools mmdetection --hyp_name FasterRCNN --aug_name Base5 --workers 6 --remove_nobbox
#python3 train.py --exp_name frcnn_albu5_newOpt --fold 4 --batch 6 --tools mmdetection --hyp_name FasterRCNN --aug_name Base5 --workers 6 --remove_nobbox

#python3 train.py --exp_name frcnn_albu0_newOpt --fold 4 --batch 6 --tools mmdetection --hyp_name FasterRCNN --aug_name Base5 --workers 6 --remove_nobbox

#python3 train.py --exp_name frcnn_albu5_baseAll --fold 4 --batch 6 --tools mmdetection --hyp_name FasterRCNN_GIoU_softNMS_OHEM --aug_name Base5  --workers 8 --remove_nobbox
#python3 train.py --exp_name swin2 --fold 4 --batch 2 --tools mmdetection --hyp_name swin2 --aug_name Base --workers 2 --remove_nobbox
#python3 train.py --exp_name swin --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base --workers 2 --remove_nobbox
#python3 train.py --exp_name swin_base5 --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base5 --workers 2 --remove_nobbox
#python3 train.py --exp_name swin_old --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base --workers 2 --remove_nobbox
python3 train.py --exp_name swin_base5 --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base5 --workers 2 --remove_nobbox
python3 train.py --exp_name swin_base_l --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base --workers 2 --remove_nobbox
python3 train.py --exp_name swin_mosaic --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name BaseMosaic --workers 2 --remove_nobbox
python3 train.py --exp_name swin_mixup --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name BaseMixUp --workers 2 --remove_nobbox