

This Code is part of my final year project . The code uses LIDAR data which and converts 3D cloud point from LIDAR data into 2D bird eye view . Then these images are trained on YOLOv4 and acuray is comapred . 


![Realtime Pizza app](readme/1.png?raw=true)
![Realtime Pizza app](readme/2.png?raw=true)
![Realtime Pizza app](readme/3.png?raw=true)
![Realtime Pizza app](readme/4.png?raw=true)


## Requirements
- CUDA >= 9.0
- Python 3
- pyquaternion, Matplotlib, PIL, numpy, cv2, tqdm (and other relevant packages which can be easily installed with pip or conda)
- PyTorch >= 1.1
  - Note that, the MGDA-related code currently can only run on PyTorch 1.1 (due to the official implementation of [MGDA](https://github.com/intel-isl/MultiObjectiveOptimization)). Such codes include `min_norm_solvers.py`, `train_single_seq_MGDA.py` and `train_multi_seq_MGDA.py`.
  
## Usage:
1. To run the code, first need to add the path to the root folder. For example:
```
export PYTHONPATH=/home/pwu/PycharmProjects/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/pwu/PycharmProjects/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
```
2. Data preparation (suppose we are now at the folder `MotionNet`):
   - Download the [nuScenes data](https://www.nuscenes.org/).
   - Run command `python data/gen_data.py --root /path/to/nuScenes/data/ --split train --savepath /path/to/the/directory/for/storing/the/preprocessed/data/`. This will generate preprocessed training data. Similarly we can prepare the validation and test data.
   - See `readme.md` in the `data` folder for more details.
3. Suppose the generated preprocessed data are in folder `/data/nuScenes_preprocessed`, then:
   - To train the model trained with spatio-temporal losses: `python train_multi_seq.py --data /data/nuScenes_preprocessed --batch 8 --nepoch 45 --nworker 4 --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --log`. This command will train the model with spatio-temporal consistency losses. See the code for more details.
   - To train the model with MGDA framework: `python train_multi_seq_MGDA.py --data /data/nuScenes_preprocessed --batch 8 --nepoch 70 --nworker 4 --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --reg_weight_cls 2.0 --log`.
   - The pre-trained model for `train_multi_seq.py` can be downloaded from [Google Drive](https://drive.google.com/file/d/1i8M4Z8VPGv-prqL5NV4pTlqtsoNu1goG/view?usp=sharing) or [Dropbox](https://www.dropbox.com/s/7f5p02d6uwfajam/model.pth?dl=0)
   - The pre-trained model for `train_multi_seq_MGDA.py` can be downloaded from [Google Drive](https://drive.google.com/file/d/1LdJferXtyC3DYBEi6zWMIUTzUQFVq0o1/view?usp=sharing) or [Dropbox](https://www.dropbox.com/s/i6arwx2zt2dagyi/model_MGDA.pth?dl=0)
   - The files `train_single_seq.py` and `train_single_seq_MGDA.py` train MotionNet exactly in the same manner, except without utilizing spatio-temporal consistency losses.
4. After obtaining the trained model, e.g., `model.pth` for `train_multi_seq.py`, we can evaluate the model performance as follows:
   - Run `python eval.py --data /path/to/the/generated/test/data --model model.pth --split test --log . --bs 1 --net MotionNet`. This will test the performance of MotionNet.

## Visualization
To visualize the results:
1. Generate the predicted results into .png images: run `python plots.py --data /path/to/nuScenes/data/ --version v1.0-trainval --modelpath model.pth --net MotionNet --nframe 10 --savepath logs`
2. Assemble the generated .png images into `.gif` or `.mp4`: `python plots.py --data /path/to/nuScenes/data/ --version v1.0-trainval --modelpath model.pth --net MotionNet --nframe 10 --savepath logs --video --format gif`


- `plots.py` contains utilities for generating the predicted images/videos.

