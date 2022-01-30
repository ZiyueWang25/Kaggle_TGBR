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
#python3 train.py --exp_name swin_base5 --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base5 --workers 2 --remove_nobbox
# python3 train.py --exp_name swin_mosaic_debug --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name BaseMosaic --workers 2 --remove_nobbox --debug
# python3 train.py --exp_name fasterrcnn_mosaic_debug --fold 4 --batch 2 --tools mmdetection --hyp_name FasterRCNN --aug_name BaseMosaic --workers 2 --remove_nobbox --debug
# python3 train.py --exp_name swin_mixup_debug --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name BaseMixUp --workers 2 --remove_nobbox --debug
#python3 train.py --exp_name swin_base_l --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base --workers 2 --remove_nobbox

# python3 train.py --exp_name swin_baseFlipRotate --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name baseFlipRotate --workers 2 --remove_nobbox
# python3 train.py --exp_name swin_baseFRHue --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name baseFRHue --workers 2 --remove_nobbox
# python3 train.py --exp_name swin_base_l --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name Base --workers 2 --remove_nobbox
# python3 train.py --exp_name swin_baseFCrop --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name baseFCrop --workers 2 --remove_nobbox
# python3 train.py --exp_name swin_baseFHue --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name baseFHue --workers 2 --remove_nobbox
# python3 train.py --exp_name swin_baseFBright --fold 4 --batch 2 --tools mmdetection --hyp_name swin --aug_name baseFBright --workers 2 --remove_nobbox

#python3 train.py --exp_name yolov5s --img_size 1280 --fold 4 --batch 16 --epochs 30 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --remove_nobbox --no_train --upload
# python3 train.py --exp_name yolov5s_cont3 --fold 4 --batch 16 --hyp_name YOLOV5\
#  --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0123_yolov5s/TGBR/0123_yolov5s/weights/best.pt\
#  --workers 8 --remove_nobbox

# python3 train.py --exp_name yolov5s_pw1d5 --fold 4 --batch 16 --hyp_name pw1d5 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_pw2 --fold 4 --batch 16 --hyp_name pw2 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_pw3 --fold 4 --batch 16 --hyp_name pw3 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_fc0d5 --fold 4 --batch 16 --hyp_name fc0d5 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_fc1 --fold 4 --batch 16 --hyp_name fc1 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_fc1d5 --fold 4 --batch 16 --hyp_name fc1d5 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_ms --multi-scale --fold 4 --batch 12 --hyp_name YOLOV5 --weights yolov5s.pt --workers 6 --remove_nobbox
# python3 train.py --exp_name yolov5s_pw1d2 --fold 4 --batch 16 --hyp_name pw1d2 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_pw0d8 --fold 4 --batch 16 --hyp_name pw0d8 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_ep50 --fold 4 --epoch 50 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --remove_nobbox

#python3 train.py --exp_name yolov5l_lr5e4 --fold 4 --batch 8 --hyp_name lr5e4 --weights yolov5l.pt --workers 6 --remove_nobbox
#python3 train.py --exp_name yolov5l_lr1e3 --fold 4 --batch 8 --hyp_name lr1e3 --weights yolov5l.pt --workers 6 --remove_nobbox

#python3 train.py --exp_name yolov5s_pre2 --img_size 768 --batch 64 --hyp_name YOLOV5 --weights yolov5s.pt --workers 12 \
#--pretrain --data_path ../data/DUO/ --debug --use-f2 False

# python3 train.py --exp_name yolov5s_wpre --fold 4 --batch 16 --hyp_name YOLOV5 \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0124_yolov5s_pre2/TGBR/0124_yolov5s_pre24/weights/best.pt \
# --workers 8 --remove_nobbox

# python3 train.py --exp_name yolov5s_lr1e3 --fold 4 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_lr1e3_plain --fold 4 --batch 16 --hyp_name YOLOV5 --weights "" --cfg yolov5s.yaml --workers 8 --remove_nobbox

# python3 train.py --exp_name yolov5s6 --fold 4 --batch 16 --hyp_name YOLOV5 --weights yolov5s6.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5l6 --fold 4 --batch 6 --hyp_name YOLOV5 --weights yolov5l6.pt --workers 4 --remove_nobbox

# python3 train.py --exp_name yolov5s_f0 --fold 0 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_f1 --fold 1 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_f2 --fold 2 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --remove_nobbox
# python3 train.py --exp_name yolov5s_f3 --fold 3 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --remove_nobbox

# python3 train.py --exp_name yolov5l_pre --img_size 768 --batch 24 --hyp_name YOLOV5 --weights yolov5l.pt --workers 12 \
# --pretrain --data_path ../data/DUO/ --debug --use-f2 False

# python3 train.py --exp_name yolov5l_wpre_f4 --fold 4 --batch 8 --hyp_name YOLOV5 \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6

# python3 train.py --exp_name yolov5l_wpre_f0 --fold 0 --batch 8 --hyp_name YOLOV5 \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6

# python3 train.py --exp_name yolov5l_wpre_f1 --fold 1 --batch 8 --hyp_name YOLOV5 \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6


# python3 train.py --exp_name yolov5s_f0_cpf --fold 0 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --use-CPF-fold
# python3 train.py --exp_name yolov5s_f1_cpf --fold 1 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --use-CPF-fold
# python3 train.py --exp_name yolov5s_f2_cpf --fold 2 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --use-CPF-fold
# python3 train.py --exp_name yolov5s_f3_cpf --fold 3 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --use-CPF-fold
# python3 train.py --exp_name yolov5s_f4_cpf --fold 4 --batch 16 --hyp_name YOLOV5 --weights yolov5s.pt --workers 8 --use-CPF-fold


# python3 train.py --exp_name yolov5l_wpre_f2 --fold 2 --batch 8 --hyp_name YOLOV5 \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6

# python3 train.py --exp_name yolov5l_wpre_f3 --fold 3 --batch 8 --hyp_name YOLOV5 \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6


# python3 train.py --exp_name yolov5l_wpre_v0 --fold 0 --batch 8 --hyp_name YOLOV5 --cv_split video_id \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6
# python3 train.py --exp_name yolov5l_wpre_v1 --fold 1 --batch 8 --hyp_name YOLOV5 --cv_split video_id \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6
# python3 train.py --exp_name yolov5l_wpre_v2 --fold 2 --batch 8 --hyp_name YOLOV5 --cv_split video_id \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6


# python3 train.py --exp_name yolov5l_wpre_v0_FP --fold 0 --batch 8 --hyp_name YOLOV5 --cv_split video_id \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6 --keep-highFP
# python3 train.py --exp_name yolov5l_wpre_v1_FP --fold 1 --batch 8 --hyp_name YOLOV5 --cv_split video_id \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6 --keep-highFP
# python3 train.py --exp_name yolov5l_wpre_v2_FP --fold 2 --batch 8 --hyp_name YOLOV5 --cv_split video_id \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0125_yolov5l_pre/TGBR/0125_yolov5l_pre5/weights/best.pt \
# --workers 6 --keep-highFP

# python3 train.py --exp_name yolov5l_v2 --cv_split v2 --batch 8 --hyp_name YOLOV5 --weights yolov5l.pt --workers 6 --debug --keep-highFP


#python3 train.py --exp_name yolov5l_v2 --cv_split v2 --batch 8 --hyp_name YOLOV5 --weights yolov5l.pt --workers 6
#python3 train.py --exp_name yolov5l_v2_test_debug --cv_split v2 --batch 8 --hyp_name YOLOV5 --workers 6 --whole_run \
#--weights /home/vincent/Kaggle/Kaggle_TGBR/output/0127_yolov5l_v2/TGBR/0127_yolov5l_v2/weights/best.pt --patience 3 --keep_nobbox --debug
# python3 train.py --exp_name yolov5l_v2_test_highFP --cv_split v2 --batch 8 --hyp_name YOLOV5 --workers 6 --whole_run \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0127_yolov5l_v2/TGBR/0127_yolov5l_v2/weights/best.pt --patience 3 --keep-highFP
# python3 train.py --exp_name yolov5l_v2_test_allData --cv_split v2 --batch 8 --hyp_name YOLOV5 --workers 6 --whole_run \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0127_yolov5l_v2/TGBR/0127_yolov5l_v2/weights/best.pt --patience 3 --keep_nobbox  --keep-highFP

# python3 train.py --exp_name yolov5l_v2_FP --cv_split v2 --batch 8 --hyp_name YOLOV5 --weights yolov5l.pt --workers 6 --keep-highFP
# python3 train.py --exp_name yolov5l_v2_FP_test_allData --cv_split v2 --batch 8 --hyp_name YOLOV5 --workers 6 --whole_run \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0128_yolov5l_v2_FP/TGBR/0128_yolov5l_v2_FP/weights/best.pt --patience 3 --keep_nobbox --keep-highFP
# python3 train.py --exp_name yolov5l_v2_FP_test_highFP --cv_split v2 --batch 8 --hyp_name YOLOV5 --workers 6 --whole_run \
# --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0128_yolov5l_v2_FP/TGBR/0128_yolov5l_v2_FP/weights/best.pt --patience 3 --keep-highFP

#python3 train.py --exp_name yolov5l_v3 --cv_split v2 --batch 8 --hyp_name YOLOV5 --weights yolov5l.pt --workers 6
#python3 train.py --exp_name yolov5l_v3_highFP --cv_split v2 --batch 8 --hyp_name YOLOV5 --weights yolov5l.pt --workers 6 --keep-highFP

# python3 train.py --exp_name yolov5l_v3_highFP_1920_debug --cv_split v2 --img_size 1800 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --debug

#python3 train.py --exp_name yolov5l_v3_highFP_whole --cv_split v2 --batch 8 --hyp_name YOLOV5 --weights yolov5l.pt --workers 6 --keep-highFP --whole_run
#python3 train.py --exp_name yolov5l_v3_highFP_1800 --cv_split v2 --img_size 1800 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP
#python3 train.py --exp_name yolov5l_v3_highFP_clahe --cv_split v2 --batch 8 --hyp_name YOLOV5 --weights yolov5l.pt --workers 6 --keep-highFP --use-clahe
# python3 train.py --exp_name yolov5l_v3_highFP_1800_e11 --cv_split v2 --img_size 1800 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --epochs 11
# python3 train.py --exp_name yolov5l_v3_highFP_1800_e11_whole --cv_split v2 --img_size 1800 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --epochs 11 --whole_run

# python3 train.py --exp_name yolov5l_v3_1800_clahe_rect_debug --cv_split v2 --img_size 2400 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --epochs 11 --use_clahe --rect --debug
# python3 train.py --exp_name yolov5l6_debug --cv_split v2 --img_size 1280 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --rect

# python3 train.py --exp_name yolov5l_v3_1800_clahe_whole --cv_split v2 --img_size 1800 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --epochs 11 --use_clahe --whole_run


#python3 train.py --exp_name yolov5l_2400_clahe_rect --cv_split v2 --img_size 2400 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --epochs 11 --use_clahe --rect
#python3 train.py --exp_name yolov5m6_2300 --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --keep-highFP --use_clahe --epochs 11
python3 train.py --exp_name yolov5m6_2300_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --whole_run
# python3 train.py --exp_name yolov5m6_3000_rect --cv_split v2 --img_size 3000 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --rect
#python3 train.py --exp_name yolov5l_2400_clahe_rect_whole --cv_split v2 --img_size 2400 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --epochs 12 --use_clahe --rect --whole_run

# python3 train.py --exp_name yolov5m6_rect_debug --cv_split v2 --img_size 3000 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --rect --debug


# python3 train.py --exp_name yolov5l_v3_1800_clahe_e16_whole --cv_split v2 --img_size 1800 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5l.pt --workers 6 --keep-highFP --epochs 15 --use_clahe --whole_run