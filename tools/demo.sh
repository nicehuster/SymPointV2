#!/usr/bin/env bash
#SBATCH -p vanke
#SBATCH -N 1
#SBATCH -J cdn-exp-l6
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem 128GB

export PYTHONPATH=./
workdir=./work_dirs/svg/svg_pointT/50p_dice5_pq_exp_nocdn_nocbl_matcher_knnpool_interp_rotate_w2x_500q_d256_layerfusion_nclsw_anglergb
workdir=./work_dirs/svg/svg_pointT/50p_dice5_pq_exp_nocdn_nocbl_matcher_knnpool_interp_rotate_w2x_500q_d256_layerfusion_nclsw_anglergb_gelu_dn_set3_noise_50p_dn2_wd_50p_cosin
workdir=./work_dirs/svg/svg_pointT/50p_dice5_pq_exp_nocdn_nocbl_matcher_knnpool_interp_rotate_w2x_500q_d256_layerfusion_nclsw_anglergb_gelu_dn_set3_noise_50p_dn2_wd_50p_cosin_1x_nofusion
workdir=./work_dirs/svg/svg_pointT/50p_dice5_pq_exp_nocdn_nocbl_matcher_knnpool_interp_rotate_w2x_500q_d256_layerfusion_nclsw_anglergb_gelu_dn_set3_noise_50p_dn2_wd_50p_cosin_2x_nofusion_nodn_svgrgb
workdir=./work_dirs/svg/svg_pointT/50p_dice5_pq_exp_nocdn_nocbl_matcher_knnpool_interp_rotate_w2x_500q_d256_layerfusion_nclsw_anglergb_gelu_dn_set3_noise_50p_dn2_wd_50p_cosin_norgb
workdir=work_dirs/svg/svg_pointT/spv2-dn2
datadir=dataset/svg/val
out=./
python tools/inference.py \
	 $workdir/svg_pointT.yaml  $workdir/epoch_50.pth \
	 --datadir $datadir \
	 --out $out 
