python multitask_classifier.py --option finetune --use_gpu\
	--output_dir result/PAL_sentbert_SICK_rotten15k_204_12\
    --epochs 15 --lr 1e-5 --lr_adapt 1e-5 --warmup_portion 0.1\
    --batch_size 16 --steps_per_epoch 2400 --eval_interval 20\
    --gradient_accumulation_step 1\
    --hidden_dropout_prob 0.1\
    --sample anneal\
    --downstream double\
    --similarity_classifier_type 'cosine-similarity'\
    --pooling_type 'mean'\
    --classification_concat_type 'add-abs'\
    --config_path config/pal_config.json