#!/bin/bash
## 'FNO'


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_ns_lr1_ar07_ds2_bs8_s16" #> "log_file/2D_ns_lr1_ar07_ds2_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_ns_lr1_ar07_ds4_bs8_s16" #> "log_file/2D_ns_lr1_ar07_ds4_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_ns_lr1_ar07_ds8_bs8_s16" #> "log_file/2D_ns_lr1_ar07_ds8_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_ns_lr1_ar07_ds16_bs8_s16" #> "log_file/2D_ns_lr1_ar07_ds16_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_ns_lr1_ar07_ds32_bs8_s16" #> "log_file/2D_ns_lr1_ar07_ds32_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds48 args.model_flmn="2D_ns_lr1_ar07_ds48_bs8_s16" #> "log_file/2D_ns_lr1_ar07_ds48_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_ns_lr1_ar07_ds64_bs8_s16" #> "log_file/2D_ns_lr1_ar07_ds64_bs8_s16.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_ns_lr1_ar07_ds2_bs8_s99" #> "log_file/2D_ns_lr1_ar07_ds2_bs8_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_ns_lr1_ar07_ds4_bs8_s99" #> "log_file/2D_ns_lr1_ar07_ds4_bs8_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_ns_lr1_ar07_ds8_bs8_s99" #> "log_file/2D_ns_lr1_ar07_ds8_bs8_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_ns_lr1_ar07_ds16_bs8_s99" #> "log_file/2D_ns_lr1_ar07_ds16_bs8_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_ns_lr1_ar07_ds32_bs8_s99" #> "log_file/2D_ns_lr1_ar07_ds32_bs8_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds48 args.model_flmn="2D_ns_lr1_ar07_ds48_bs8_s99" #> "log_file/2D_ns_lr1_ar07_ds48_bs8_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_ns_lr1_ar07_ds64_bs8_s99" #> "log_file/2D_ns_lr1_ar07_ds64_bs8_s99.txt" 2#>&1 

# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_ns_ts_down_ds2_bs8_s16" > "log_file/2D_ns_ts_down_ds2_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_ns_ts_down_ds4_bs8_s16" > "log_file/2D_ns_ts_down_ds4_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_ns_ts_down_ds8_bs8_s16" > "log_file/2D_ns_ts_down_ds8_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_ns_ts_down_ds16_bs8_s16" > "log_file/2D_ns_ts_down_ds16_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_ns_ts_down_ds32_bs8_s16" > "log_file/2D_ns_ts_down_ds32_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds48 args.model_flmn="2D_ns_ts_down_ds48_bs8_s16" > "log_file/2D_ns_ts_down_ds48_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_ns_ts_down_ds64_bs8_s16" > "log_file/2D_ns_ts_down_ds64_bs8_s16.txt" 2>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_ns_ts_down_ds2_bs8_s99" > "log_file/2D_ns_ts_down_ds2_bs8_s99.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_ns_ts_down_ds4_bs8_s99" > "log_file/2D_ns_ts_down_ds4_bs8_s99.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_ns_ts_down_ds8_bs8_s99" > "log_file/2D_ns_ts_down_ds8_bs8_s99.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_ns_ts_down_ds16_bs8_s99" > "log_file/2D_ns_ts_down_ds16_bs8_s99.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_ns_ts_down_ds32_bs8_s99" > "log_file/2D_ns_ts_down_ds32_bs8_s99.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds48 args.model_flmn="2D_ns_ts_down_ds48_bs8_s99" > "log_file/2D_ns_ts_down_ds48_bs8_s99.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_ns_ts_down_ds64_bs8_s99" > "log_file/2D_ns_ts_down_ds64_bs8_s99.txt" 2>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds2 args.FNO_model_flmn="2D_ns_fno_all_ds2_bs16_s16" #> "log_file/2D_ns_fno_all_ds2_bs16_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds4 args.FNO_model_flmn="2D_ns_fno_all_ds4_bs16_s16" #> "log_file/2D_ns_fno_all_ds4_bs16_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds8 args.FNO_model_flmn="2D_ns_fno_all_ds8_bs16_s16" #> "log_file/2D_ns_fno_all_ds8_bs16_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds16 args.FNO_model_flmn="2D_ns_fno_all_ds16_bs16_s16" #> "log_file/2D_ns_fno_all_ds16_bs16_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds32 args.FNO_model_flmn="2D_ns_fno_all_ds32_bs16_s16" #> "log_file/2D_ns_fno_all_ds32_bs16_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds48 args.FNO_model_flmn="2D_ns_fno_all_ds48_bs16_s16" #> "log_file/2D_ns_fno_all_ds48_bs16_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds64 args.FNO_model_flmn="2D_ns_fno_all_ds64_bs16_s16" #> "log_file/2D_ns_fno_all_ds64_bs16_s16.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds2 args.FNO_model_flmn="2D_ns_fno_all_ds2_bs16_s99" #> "log_file/2D_ns_fno_all_ds2_bs16_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds4 args.FNO_model_flmn="2D_ns_fno_all_ds4_bs16_s99" #> "log_file/2D_ns_fno_all_ds4_bs16_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds8 args.FNO_model_flmn="2D_ns_fno_all_ds8_bs16_s99" #> "log_file/2D_ns_fno_all_ds8_bs16_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds16 args.FNO_model_flmn="2D_ns_fno_all_ds16_bs16_s99" #> "log_file/2D_ns_fno_all_ds16_bs16_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds32 args.FNO_model_flmn="2D_ns_fno_all_ds32_bs16_s99" #> "log_file/2D_ns_fno_all_ds32_bs16_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds48 args.FNO_model_flmn="2D_ns_fno_all_ds48_bs16_s99" #> "log_file/2D_ns_fno_all_ds48_bs16_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds64 args.FNO_model_flmn="2D_ns_fno_all_ds64_bs16_s99" #> "log_file/2D_ns_fno_all_ds64_bs16_s99.txt" 2#>&1 

CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds2 args.FNO_model_flmn="2D_ns_fno_lie_ds2_bs16_s16" > "log_file/2D_ns_fno_lie_ds2_bs16_s16.txt" 2>&1 
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds4 args.FNO_model_flmn="2D_ns_fno_lie_ds4_bs16_s16" > "log_file/2D_ns_fno_lie_ds4_bs16_s16_2.txt" 2>&1 
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds8 args.FNO_model_flmn="2D_ns_fno_lie_ds8_bs16_s16" > "log_file/2D_ns_fno_lie_ds8_bs16_s16.txt" 2>&1 
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds16 args.FNO_model_flmn="2D_ns_fno_lie_ds16_bs16_s16" > "log_file/2D_ns_fno_lie_ds16_bs16_s16.txt" 2>&1 
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds32 args.FNO_model_flmn="2D_ns_fno_lie_ds32_bs16_s16" > "log_file/2D_ns_fno_lie_ds32_bs16_s16.txt" 2>&1 
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds48 args.FNO_model_flmn="2D_ns_fno_lie_ds48_bs16_s16" > "log_file/2D_ns_fno_lie_ds48_bs16_s16.txt" 2>&1 
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds64 args.FNO_model_flmn="2D_ns_fno_lie_ds64_bs16_s16" > "log_file/2D_ns_fno_lie_ds64_bs16_s16.txt" 2>&1 

# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_ns_aux_lie_ds2_bs8_s16" > "log_file/2D_ns_aux_lie_ds2_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_ns_aux_lie_ds4_bs8_s16" > "log_file/2D_ns_aux_lie_ds4_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_ns_aux_lie_ds8_bs8_s16" > "log_file/2D_ns_aux_lie_ds8_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_ns_aux_lie_ds16_bs8_s16" > "log_file/2D_ns_aux_lie_ds16_bs8_s16.txt" 2>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_ns_aux_lie_ds32_bs8_s16" #> "log_file/2D_ns_aux_lie_ds32_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds48 args.model_flmn="2D_ns_aux_lie_ds48_bs8_s16" #> "log_file/2D_ns_aux_lie_ds48_bs8_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_ns_aux_lie_ds64_bs8_s16" #> "log_file/2D_ns_aux_lie_ds64_bs8_s16.txt" 2#>&1 

