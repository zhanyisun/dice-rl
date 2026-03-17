## Data processing scripts

To generate data for training and finetuning, first download raw data from [this link](https://huggingface.co/datasets/wintermelontree/raw_robomimic_data/tree/main) or the official Robomimic repository. Then use the following commands to generate processed data for pretraining. 

Can (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/can-img/ph_pretrain --normalize
```

Can (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/can-img/ph_pretrain --normalize --cameras agentview robot0_eye_in_hand
```

Square (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/square/ph_pretrain --normalize
```

Square (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/square/ph_pretrain --normalize --cameras agentview robot0_eye_in_hand
```

Transport (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/transport-img/ph_pretrain --normalize
```     

Transport (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/transport-img/ph_pretrain --normalize --cameras robot0_eye_in_hand robot1_eye_in_hand shouldercamera0 shouldercamera1 
``` 

Tool Hang (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/tool_hang-img/ph_pretrain --normalize
```

Tool Hang (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/tool_hang-img/ph_pretrain --normalize --cameras sideview  robot0_eye_in_hand
```

By default, DICE-RL uses RLPD for finetuning. To generate data for finetuning, simply add `--truncate` to the command used for pretraining, which will truncate the trajectories to have exactly one success at the end. This is to ensure the value learning between offline data and online data is consistent.
