export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
exp_id=exp

######## method
method=EWC

######## incremental task num
shuffle_seed=-1
sequence_seed=-1

######## elastic weights consolidation
online=True
ewc_loss_coef=10.0
sample_size=1024

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features (You need to modify the feature path to your own.)
feat_root=data/qvhight_features/features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
for task_num in 2 5 10 20
do
    case $task_num in
      2|5) bsz=32 ;;  # task_num = 2 or 5
      *)   bsz=16 ;;  # else
    esac
    
    PYTHONPATH=$PYTHONPATH:. python moment_detr_CL/EWC.py \
    --method ${method} \
    --task_num ${task_num} \
    --shuffle_seed ${shuffle_seed} \
    --sequence_seed ${sequence_seed} \
    --online ${online} \
    --ewc_loss_coef ${ewc_loss_coef} \
    --sample_size ${sample_size} \
    --dset_name ${dset_name} \
    --ctx_mode ${ctx_mode} \
    --train_path ${train_path} \
    --eval_path ${eval_path} \
    --eval_split_name ${eval_split_name} \
    --v_feat_dirs ${v_feat_dirs[@]} \
    --v_feat_dim ${v_feat_dim} \
    --t_feat_dir ${t_feat_dir} \
    --t_feat_dim ${t_feat_dim} \
    --bsz ${bsz} \
    --results_root ${results_root} \
    --exp_id ${exp_id} \
    ${@:1}
done