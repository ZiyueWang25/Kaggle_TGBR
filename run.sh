
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
# python3 train.py --exp_name yolov5m6_2300_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --whole_run

# python3 train.py --exp_name yolov5m6_2300_sheep_cnt --cv_split v2 --img_size 2300 --batch 4 --hyp_name sheep\
#  --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0130_yolov5m6_2300_sheep/TGBR/0130_yolov5m6_2300_sheep/weights/last.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --optimizer SGD

# python3 train.py --exp_name yolov5m6_noClahe_2300_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --keep-highFP --epochs 11 --whole_run
# python3 train.py --exp_name yolov5m6_2300_sheep_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name sheep --weights yolov5m6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --optimizer SGD --whole_run


# python3 train.py --exp_name yolov5m6_MU5_2300 --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4_MU5 --weights yolov5m6.pt --workers 6 --keep-highFP --use_clahe --epochs 15 

# python3 train.py --exp_name yolov5m6_noClahe_2300 --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --keep-highFP --epochs 11

# python3 train.py --exp_name yolov5m6_noHighFP_2300 --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --epochs 11 --use_clahe

# python3 train.py --exp_name yolov5m6_noClahe_noHighFP_2300 --cv_split v2 --img_size 2300 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5m6.pt --workers 6 --epochs 11

# python3 train.py --exp_name yolov5s6_debug --cv_split v2 --img_size 3100 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5s6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --debug

# python3 train.py --exp_name yolov5s6_3100 --cv_split v2 --img_size 3100 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5s6.pt --workers 6 --keep-highFP --use_clahe --epochs 11
# python3 train.py --exp_name yolov5s6_3100_whole --cv_split v2 --img_size 3100 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5s6.pt --workers 6 --keep-highFP --use_clahe --epochs 11 --whole_run


# python3 train.py --exp_name yolov5s6_3100_noClahe_sheep_whole --cv_split v2 --img_size 3100 --batch 4 --hyp_name sheep --weights yolov5s6.pt --workers 6 --keep-highFP --epochs 11 --optimizer SGD

# python3 train.py --exp_name frcnn_1600_highFP --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name FasterRCNN --epochs 11 --workers 6 --aug_name Base5 --keep-highFP 

# python3 train.py --exp_name vf_debug --img_size 1700 --batch 4 --cv_split v2 --tools mmdetection --hyp_name vfnet --epochs 8 --workers 6 --aug_name Base5 --keep-highFP --debug


# python3 train.py --exp_name frcnn_1600_highFP_e8 --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name FasterRCNN --epochs 8 --workers 4 --aug_name Base5 --keep-highFP

# python3 train.py --exp_name frcnn_1600_highFP_e8_wholeRun --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name FasterRCNN --epochs 8 --workers 4 --aug_name Base5 --keep-highFP --whole_run

# python3 train.py --exp_name vf_1700_highFP_e11 --img_size 1700 --batch 4 --cv_split v2 --tools mmdetection --hyp_name vfnet --epochs 11 --workers 4 --aug_name Base5 --keep-highFP

# python3 train.py --exp_name m6_1200_sliced_S720_ma025 --img_size 1200  --batch 14 --hyp_name YOLOV5 --epochs 15 --workers 8 --weights yolov5m6.pt --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/MA0.25/

# python3 train.py --exp_name m6_1200_sliced_S720_ma025_lr5e4 --img_size 1200  --batch 14 --hyp_name YOLOV5_LR5e4 --epochs 15 --workers 8 --weights yolov5m6.pt --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/MA0.25/


# python3 train.py --exp_name vf_1700_highFP_e11_whole --img_size 1700 --batch 4 --cv_split v2 --tools mmdetection --hyp_name vfnet --epochs 11 --workers 4 --aug_name Base5 --keep-highFP --whole_run 

#python3 train.py --exp_name m6_1200_sliced_S720_ma025_lr5e4_whole --img_size 1200  --batch 14 --hyp_name YOLOV5_LR5e4 --epochs 18 --workers 8 --weights yolov5m6.pt --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/whole_S720xS720_MA0.25/ --whole_run

# python3 train.py --exp_name m6_1200_sliced_S720_ma025_lr5e4_cnt20 --img_size 1200  --batch 14 --hyp_name YOLOV5_LR5e4 --epochs 20 --workers 8 --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/MA0.25/ --weights /home/vincent/Kaggle/Kaggle_TGBR/output/0203_m6_1200_sliced_S720_ma025_lr5e4/TGBR/0203_m6_1200_sliced_S720_ma025_lr5e4/weights/last.pt --workers 6 --score_thres 0.3

# python3 train.py --exp_name m6_1200_sliced_S720_ma025_lr5e4_cnt20 --img_size 1200  --batch 14 --hyp_name YOLOV5_SA_LR5e4 --epochs 30 --workers 8 --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/MA0.25/ --weights yolov5m6.pt --workers 6 --score_thres 0.3


# python3 train.py --exp_name m6_1000_sliced_S400x711_ma025 --img_size 1024  --batch 20 --hyp_name YOLOV5 --epochs 20 --workers 8 --weights yolov5m6.pt --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/S400xS711_MA0.25/ --score_thres 0.3

# python3 train.py --exp_name m6_1000_sliced_S400x711_ma025_LR5e4 --img_size 1024  --batch 20 --hyp_name YOLOV5_LR5e4 --epochs 20 --workers 8 --weights yolov5m6.pt --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/S400xS711_MA0.25/ --score_thres 0.3

# python3 train.py --exp_name yolov5x6_1320 --cv_split v2 --img_size 1320 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5x6.pt --workers 4 --keep-highFP --epochs 20

# python3 train.py --exp_name yolov5x6_1320_MU8 --cv_split v2 --img_size 1320 --batch 4 --hyp_name YOLOV5_B4_MU8 --weights yolov5x6.pt --workers 4 --keep-highFP --epochs 20


# python3 train.py --exp_name yolov5x6_1320_debug --cv_split v2 --img_size 1320 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5x6.pt --workers 4 --keep-highFP --epochs 15 --debug

# python3 train.py --exp_name yolov5x6_1320_whole --cv_split v2 --img_size 1320 --batch 4 --hyp_name YOLOV5_B4 --weights yolov5x6.pt --workers 4 --keep-highFP --epochs 15 --whole_run


# python3 train.py --exp_name yolov5s6_debug --cv_split v2 --img_size 3072 --batch 4 --hyp_name HIGH --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12 --debug
# python3 train.py --exp_name yolov5s6_B --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12
# python3 train.py --exp_name yolov5s6_MA --cv_split v2 --img_size 3072 --batch 4 --hyp_name MED --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12
# python3 train.py --exp_name yolov5s6_HA --cv_split v2 --img_size 3072 --batch 4 --hyp_name HIGH --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12

# python3 train.py --exp_name yolov5m6_B --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12

#python3 train.py --exp_name yolov5m_TPH_B_debug --cv_split v2 --img_size 1900 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml --debug

# python3 train.py --exp_name m_TPH_1900_B --cv_split v2 --img_size 1900 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml
# python3 train.py --exp_name m_TPH_1900_B_whole --cv_split v2 --img_size 1900 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml --whole_run

# python3 train.py --exp_name m_P7_B_debug --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/hub/yolov5m-p7.yaml --debug

# python3 train.py --exp_name m_P7_2300_B --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/hub/yolov5m-p7.yaml
# python3 train.py --exp_name m_TPH_1900_B_LR1e3 --cv_split v2 --img_size 1900 --batch 4 --hyp_name B_LR1e3 --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml

# python3 train.py --exp_name yolov5m6_B_LS02 --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2
# python3 train.py --exp_name yolov5m6_B_LS06 --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.6
# python3 train.py --exp_name m_P7_2300_B_LR1e3 --cv_split v2 --img_size 2300 --batch 4 --hyp_name B_LR1e3 --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/hub/yolov5m-p7.yaml
# python3 train.py --exp_name yolov5s6_B_whole --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12 --whole_run

# python3 train.py --exp_name yolov5m6_B_LS02_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --whole_run
#python3 train.py --exp_name yolov5m6_B_LS01 --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.1
#python3 train.py --exp_name m_TPH_1900_B_LS02 --cv_split v2 --img_size 1900 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml --label-smoothing 0.2

# python3 train.py --exp_name yolov5m6_B_LS02_newGT_debug --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --debug

# python3 train.py --exp_name yolov5m6_B_LS02_newGT --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot

# python3 train.py --exp_name yolov5s6_B_LS02_newGT --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot

# python3 train.py --exp_name yolov5m6_B_LS02_newGT --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --whole_run


# python3 train.py --exp_name m_TPH_1900_B_newGT --cv_split v2 --img_size 1900 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml --use_new_annot

# python3 train.py --exp_name yolov5m6_B_LS02_newGT_clahe --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --use_clahe


# python3 train.py --exp_name yolov5m6_B_LS02_newGT_CV_video1 --cv_split video_id --fold 1 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot
# python3 train.py --exp_name yolov5m6_B_LS02_newGT_clahe_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --use_clahe --whole_run


# python3 train.py --exp_name m6_B_LS02_newGT_clahe --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_clahe --use_new_annot 
# python3 train.py --exp_name m6_B_LS02_newGT_clahe_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_clahe --use_new_annot --whole_run --upload

# python3 train.py --exp_name s6_B_LS02_newGT --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot
# python3 train.py --exp_name s6_B_LS02_newGT_whole --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --whole_run --upload

# python3 train.py --exp_name m_TPH_1900_B_LS02_newGT --cv_split v2 --img_size 1870 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml --use_new_annot --label-smoothing 0.2 
# python3 train.py --exp_name m_TPH_1900_B_LS02_newGT_whole --cv_split v2 --img_size 1900 --batch 4 --hyp_name Base --weights yolov5m.pt --workers 2 --keep-highFP --epochs 12 --cfg /home/vincent/Kaggle/Kaggle_TGBR/yolov5/models/yolov5m-xs-tph.yaml --use_new_annot --label-smoothing 0.2  --whole_run --upload

# python3 train.py --exp_name s6_B_LS02_newGT_clahe --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --use_clahe
# python3 train.py --exp_name s6_B_LS02_newGT_clahe_whole --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --use_clahe --whole_run --upload

# mmdetection
# 0202_frcnn_1600_highFP_e8_wholerun is incorrect, need to fix

# python3 train.py --exp_name m6_B_LS02_newLGBT --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot 
# python3 train.py --exp_name m6_B_LS02_newLGBT_whole --cv_split v2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --whole_run --upload

# python3 train.py --exp_name s6_B_LS02_newGT_imgxxx_debug --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 10 --label-smoothing 0.2 --use_new_annot --copy_image
# python3 train.py --exp_name s6_B_LS02_newGT_imgxxx_whole --cv_split v2 --img_size 3072 --batch 4 --hyp_name Base --weights yolov5s6.pt --workers 2 --keep-highFP --epochs 10 --label-smoothing 0.2 --use_new_annot --whole_run

# python3 train.py --exp_name m6_B_LS02_newLGBT_video0 --cv_split video_id --fold 0 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot --resume
# python3 train.py --exp_name m6_B_LS02_newLGBT_video2 --cv_split video_id --fold 2 --img_size 2300 --batch 4 --hyp_name Base --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 12 --label-smoothing 0.2 --use_new_annot 


# python3 -m torch.distributed.launch --nproc_per_node 2  train.py --exp_name m6_B_LS02_LGBT_newP --cv_split v2 --img_size 2300 --batch 4 --optimizer Adam --hyp_name NEW --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 30 --label-smoothing 0.2 --use_new_annot --resume

# python3 -m torch.distributed.launch --nproc_per_node 2  train.py --exp_name m6_B_LS02_LGBT_newP_whole --cv_split v2 --img_size 2300 --batch 4 --optimizer Adam --hyp_name NEW --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 22 --label-smoothing 0.2 --use_new_annot --whole_run

# python3 -m torch.distributed.launch --nproc_per_node 2  train.py --exp_name m6_B_LS02_LGBT_newP_clahe_whole --cv_split v2 --img_size 2300 --batch 4 --optimizer Adam --hyp_name NEW --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 22 --label-smoothing 0.2 --use_new_annot --whole_run


# python3 train.py --exp_name crcnn_1600 --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 11 --workers 6 --aug_name Base5 --keep-highFP --use_new_annot

# python3 -m torch.distributed.launch --nproc_per_node 2  train.py --exp_name m6_B_LS02_LGBT_newP_clahe --cv_split v2 --img_size 2300 --batch 4 --optimizer Adam --hyp_name NEW --weights yolov5m6.pt --workers 2 --keep-highFP --epochs 20 --label-smoothing 0.2 --use_new_annot --use_clahe --local_rank 0

# python3 train.py --exp_name crcnn_OHEM_softNMS --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 11 --workers 6 --aug_name Base5 --keep-highFP --use_new_annot

# python3 train.py --exp_name crcnn_1600_whole --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 11 --workers 6 --aug_name Base5 --keep-highFP --use_new_annot --whole_run

# python3 train.py --exp_name crcnn_1600_baseFlip --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 10 --workers 6 --aug_name baseFlipRotate --keep-highFP --use_new_annot

# python3 train.py --exp_name crcnn_1600_base7 --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 10 --workers 6 --aug_name Base7 --keep-highFP --use_new_annot

# python3 train.py --exp_name crcnn_1600_baseFlip_whole --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 10 --workers 6 --aug_name baseFlipRotate --keep-highFP --use_new_annot --whole

# python3 train.py --exp_name crcnn_1600_e20 --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 20 --workers 6 --aug_name Base7 --keep-highFP --use_new_annot

# python3 train.py --exp_name crcnn_1600_base7_whole --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 11 --workers 6 --aug_name Base7 --keep-highFP --use_new_annot --whole_run


# python3 train.py --exp_name crcnn_1600_e20 --img_size 1600 --batch 4 --cv_split v2 --tools mmdetection --hyp_name CRCNN --epochs 20 --workers 6 --aug_name Base85 --keep-highFP --use_new_annot

# python3 train.py --exp_name m6_1200_sliced_S720_ma025_lr5e3 --img_size 1200  --batch 14 --hyp_name YOLOV5_LR5e3 --epochs 15 --workers 8 --weights yolov5m6.pt --sliced --data_path /home/vincent/Kaggle/data/tensorflow-great-barrier-reef/sliced/MA0.25/
