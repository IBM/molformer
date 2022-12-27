python finetune_pubchem_light_classification.py \
        --device cuda \
        --batch_size 32  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 8\
        --max_epochs 500 \
        --num_feats 32 \
        --seed_path '../data/checkpoints/linear_model.ckpt' \
        --dataset_name hiv \
        --data_root ../data/hiv \
        --measure_name HIV_active \
        --dims 768 768 768 1 \
        --checkpoints_folder './checkpoints_hiv'\
        --num_classes 2 \