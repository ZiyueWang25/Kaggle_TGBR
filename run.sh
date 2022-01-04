python3 train.py --exp_name yolov5s_fold0 --fold 0 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10 --copy_image
python3 train.py --exp_name yolov5s_fold1 --fold 1 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10
python3 train.py --exp_name yolov5s_fold2 --fold 2 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10
python3 train.py --exp_name yolov5s_fold3 --fold 3 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10
python3 train.py --exp_name yolov5s_fold4 --fold 4 --remove_nobbox --batch 16 --epochs 20 --weights yolov5s.pt --workers 10