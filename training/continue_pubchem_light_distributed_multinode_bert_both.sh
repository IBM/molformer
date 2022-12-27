source /opt/share/anaconda3-2019.03/x86_64/etc/profile.d/conda.sh
conda activate /dccstor/bmbelgod1/environments/MolTran_CUDA11
python train_pubchem_light.py \
        --device cuda \
        --n_batch 800  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --max_len 202 \
        --d_dropout 0.2 \
        --lr_start 3e-5 \
        --lr_multiplier 8 \
        --n_workers 16 \
        --max_epochs 4 \
        --gpu 8 \
        --num_nodes 2 \
        --accelerator ddp \
        --num_feats 32 \
        --root_dir . \
        --checkpoint_every 1000 \
        --grad_acc 1\
        --train_load 'both' \
        --eval_every 2500 \
        --rotate \
        --debug \
        --model_arch 'BERT_16GPU_Long_Run_with_Rotate_Continued' \
        --restart_path /dccstor/bmbelgod1/projects/MolTran/lightning_logs/version_11/checkpoints/epoch\=2-step\=178277.ckpt \
        | tee $HOSTNAME.$LSF_PM_XPROCID.$(date +%F_%R).log
        #--restart_path /dccstor/bmbelgod1/projects/MolTran/lightning_logs/version_11/checkpoints/N-Step-Checkpoint_2_180000.ckpt \
