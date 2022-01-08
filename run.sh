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

python3 train.py --exp_name yolov5s_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name Base --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50

python3 train.py --exp_name yolov5s_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name LR0005 --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50

python3 train.py --exp_name yolov5s_fold0_Base_newcv_nobbox --fold 0 --batch 16 --epochs 200 --hyp_name LRF001 --weights yolov5s.pt --workers 12 --remove_nobbox --patience 50