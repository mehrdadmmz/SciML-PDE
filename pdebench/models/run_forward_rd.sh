#!/bin/bash
## 'FNO'

# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_lr1_ar07_ds2_bs2_s16" #> "log_file/2D_rd_lr1_ar07_ds2_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_lr1_ar07_ds4_bs2_s16" #> "log_file/2D_rd_lr1_ar07_ds4_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_lr1_ar07_ds8_bs2_s16" #> "log_file/2D_rd_lr1_ar07_ds8_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_lr1_ar07_ds16_bs2_s16" #> "log_file/2D_rd_lr1_ar07_ds16_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_lr1_ar07_ds32_bs2_s16" #> "log_file/2D_rd_lr1_ar07_ds32_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_lr1_ar07_ds64_bs2_s16" #> "log_file/2D_rd_lr1_ar07_ds64_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_lr1_ar07_ds128_bs2_s16" #> "log_file/2D_rd_lr1_ar07_ds128_bs2_s16.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_lr1_ar07_ds2_bs2_s99" #> "log_file/2D_rd_lr1_ar07_ds2_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_lr1_ar07_ds4_bs2_s99" #> "log_file/2D_rd_lr1_ar07_ds4_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_lr1_ar07_ds8_bs2_s99" #> "log_file/2D_rd_lr1_ar07_ds8_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_lr1_ar07_ds16_bs2_s99" #> "log_file/2D_rd_lr1_ar07_ds16_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_lr1_ar07_ds32_bs2_s99" #> "log_file/2D_rd_lr1_ar07_ds32_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_lr1_ar07_ds64_bs2_s99" #> "log_file/2D_rd_lr1_ar07_ds64_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_lr1_ar07_ds128_bs2_s99" #> "log_file/2D_rd_lr1_ar07_ds128_bs2_s99.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_lr1_ar05_ds2_bs2_s99" #> "log_file/2D_rd_lr1_ar05_ds2_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_lr1_ar05_ds4_bs2_s99" #> "log_file/2D_rd_lr1_ar05_ds4_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_lr1_ar05_ds8_bs2_s99" #> "log_file/2D_rd_lr1_ar05_ds8_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_lr1_ar05_ds16_bs2_s99" #> "log_file/2D_rd_lr1_ar05_ds16_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_lr1_ar05_ds32_bs2_s99" #> "log_file/2D_rd_lr1_ar05_ds32_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_lr1_ar05_ds64_bs2_s99" #> "log_file/2D_rd_lr1_ar05_ds64_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_lr1_ar05_ds128_bs2_s99" #> "log_file/2D_rd_lr1_ar05_ds128_bs2_s99.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_lr1_ar1_ds2_bs2_s99" #> "log_file/2D_rd_lr1_ar1_ds2_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_lr1_ar1_ds4_bs2_s99" #> "log_file/2D_rd_lr1_ar1_ds4_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_lr1_ar1_ds8_bs2_s99" #> "log_file/2D_rd_lr1_ar1_ds8_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_lr1_ar1_ds16_bs2_s99" #> "log_file/2D_rd_lr1_ar1_ds16_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_lr1_ar1_ds32_bs2_s99" #> "log_file/2D_rd_lr1_ar1_ds32_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_lr1_ar1_ds64_bs2_s99" #> "log_file/2D_rd_lr1_ar1_ds64_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_lr1_ar1_ds128_bs2_s99" #> "log_file/2D_rd_lr1_ar1_ds128_bs2_s99.txt" 2#>&1 

# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_ts_down_ds2_bs2_s99" #> "log_file/2D_rd_ts_down_ds2_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_ts_down_ds4_bs2_s99" #> "log_file/2D_rd_ts_down_ds4_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_ts_down_ds8_bs2_s99" #> "log_file/2D_rd_ts_down_ds8_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_ts_down_ds16_bs2_s99" #> "log_file/2D_rd_ts_down_ds16_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_ts_down_ds32_bs2_s99" #> "log_file/2D_rd_ts_down_ds32_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_ts_down_ds64_bs2_s99" #> "log_file/2D_rd_ts_down_ds64_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_ts_down_ds128_bs2_s99" #> "log_file/2D_rd_ts_down_ds128_bs2_s99.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_ts_down_ds2_bs2_s16" #> "log_file/2D_rd_ts_down_ds2_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_ts_down_ds4_bs2_s16" #> "log_file/2D_rd_ts_down_ds4_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_ts_down_ds8_bs2_s16" #> "log_file/2D_rd_ts_down_ds8_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_ts_down_ds16_bs2_s16" #> "log_file/2D_rd_ts_down_ds16_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_ts_down_ds32_bs2_s16" #> "log_file/2D_rd_ts_down_ds32_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_ts_down_ds64_bs2_s16" #> "log_file/2D_rd_ts_down_ds64_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_ts_down_ds128_bs2_s16" #> "log_file/2D_rd_ts_down_ds128_bs2_s16.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_tsdecomp_down_ds2_bs2_s16" #> "log_file/2D_rd_tsdecomp_down_ds2_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_tsdecomp_down_ds4_bs2_s16" #> "log_file/2D_rd_tsdecomp_down_ds4_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_tsdecomp_down_ds8_bs2_s16" #> "log_file/2D_rd_tsdecomp_down_ds8_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_tsdecomp_down_ds16_bs2_s16" #> "log_file/2D_rd_tsdecomp_down_ds16_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_tsdecomp_down_ds32_bs2_s16" #> "log_file/2D_rd_tsdecomp_down_ds32_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_tsdecomp_down_ds64_bs2_s16" #> "log_file/2D_rd_tsdecomp_down_ds64_bs2_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_tsdecomp_down_ds128_bs2_s16" #> "log_file/2D_rd_tsdecomp_down_ds128_bs2_s16.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_tsdecomp_down_ds2_bs2_s99" #> "log_file/2D_rd_tsdecomp_down_ds2_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_tsdecomp_down_ds4_bs2_s99" #> "log_file/2D_rd_tsdecomp_down_ds4_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_tsdecomp_down_ds8_bs2_s99" #> "log_file/2D_rd_tsdecomp_down_ds8_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_tsdecomp_down_ds16_bs2_s99" #> "log_file/2D_rd_tsdecomp_down_ds16_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_tsdecomp_down_ds32_bs2_s99" #> "log_file/2D_rd_tsdecomp_down_ds32_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_tsdecomp_down_ds64_bs2_s99" #> "log_file/2D_rd_tsdecomp_down_ds64_bs2_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_tsdecomp_down_ds128_bs2_s99" #> "log_file/2D_rd_tsdecomp_down_ds128_bs2_s99.txt" 2#>&1 

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds2 args.model_flmn="2D_rd_tsdecomp_down_ds2_bs2_s17" #> "log_file/2D_rd_tsdecomp_down_ds2_bs2_s17.txt" 2#>&1 
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds4 args.model_flmn="2D_rd_tsdecomp_down_ds4_bs2_s17" #> "log_file/2D_rd_tsdecomp_down_ds4_bs2_s17.txt" 2#>&1 
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds8 args.model_flmn="2D_rd_tsdecomp_down_ds8_bs2_s17" #> "log_file/2D_rd_tsdecomp_down_ds8_bs2_s17.txt" 2#>&1 
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds16 args.model_flmn="2D_rd_tsdecomp_down_ds16_bs2_s17" #> "log_file/2D_rd_tsdecomp_down_ds16_bs2_s17.txt" 2#>&1 
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds32 args.model_flmn="2D_rd_tsdecomp_down_ds32_bs2_s17" #> "log_file/2D_rd_tsdecomp_down_ds32_bs2_s17.txt" 2#>&1 
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds64 args.model_flmn="2D_rd_tsdecomp_down_ds64_bs2_s17" #> "log_file/2D_rd_tsdecomp_down_ds64_bs2_s17.txt" 2#>&1 
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_aux_forward.py dataset=basic_ds128 args.model_flmn="2D_rd_tsdecomp_down_ds128_bs2_s17" #> "log_file/2D_rd_tsdecomp_down_ds128_bs2_s17.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds2 args.FNO_model_flmn="2D_rd_fno_all_ds2_bs4_s16" #> "log_file/2D_rd_fno_all_ds2_bs4_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds4 args.FNO_model_flmn="2D_rd_fno_all_ds4_bs4_s16" #> "log_file/2D_rd_fno_all_ds4_bs4_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds8 args.FNO_model_flmn="2D_rd_fno_all_ds8_bs4_s16" #> "log_file/2D_rd_fno_all_ds8_bs4_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds16 args.FNO_model_flmn="2D_rd_fno_all_ds16_bs4_s16" #> "log_file/2D_rd_fno_all_ds16_bs4_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds32 args.FNO_model_flmn="2D_rd_fno_all_ds32_bs4_s16" #> "log_file/2D_rd_fno_all_ds32_bs4_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds64 args.FNO_model_flmn="2D_rd_fno_all_ds64_bs4_s16" #> "log_file/2D_rd_fno_all_ds64_bs4_s16.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds128 args.FNO_model_flmn="2D_rd_fno_all_ds128_bs4_s16" #> "log_file/2D_rd_fno_all_ds128_bs4_s16.txt" 2#>&1 


# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds2 args.FNO_model_flmn="2D_rd_fno_all_ds2_bs4_s99" #> "log_file/2D_rd_fno_all_ds2_bs4_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds4 args.FNO_model_flmn="2D_rd_fno_all_ds4_bs4_s99" #> "log_file/2D_rd_fno_all_ds4_bs4_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds8 args.FNO_model_flmn="2D_rd_fno_all_ds8_bs4_s99" #> "log_file/2D_rd_fno_all_ds8_bs4_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds16 args.FNO_model_flmn="2D_rd_fno_all_ds16_bs4_s99" #> "log_file/2D_rd_fno_all_ds16_bs4_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds32 args.FNO_model_flmn="2D_rd_fno_all_ds32_bs4_s99" #> "log_file/2D_rd_fno_all_ds32_bs4_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds64 args.FNO_model_flmn="2D_rd_fno_all_ds64_bs4_s99" #> "log_file/2D_rd_fno_all_ds64_bs4_s99.txt" 2#>&1 
# CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train_models_forward.py dataset=basic_ds128 args.FNO_model_flmn="2D_rd_fno_all_ds128_bs4_s99" #> "log_file/2D_rd_fno_all_ds128_bs4_s99.txt" 2#>&1 

