
python -u run_longExp.py \
      --is_training 0 \
      --root_path ./dataset/new_data6 \
      --data_path 31022519770825021X.csv \
      --model_id new_rehab_wb6 \
      --model PatchTST \
      --data rehab \
      --itr 1 \
      --train_epochs 1 \
      --lradj 'constant' \
      --seq_len 7 \
      --pred_len 7 \
      --batch_size 4 \
      --patch_len 3 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 3 \
      --learning_rate 0.0001
      
python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/new_data6 \
      --data_path 31022519770825021X.csv \
      --model_id new_rehab_wb6 \
      --model Autoformer \
      --data rehab \
      --itr 1 \
      --train_epochs 100 \
      --lradj 'constant' \
      --seq_len 7 \
      --pred_len 7 \
      --batch_size 4 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 3 \
      --enc_in 6 \
      --dec_in 6 \
      --c_out 6
      
python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/new_data6 \
      --data_path 31022519770825021X.csv \
      --model_id new_rehab_wb6 \
      --model Informer \
      --data rehab \
      --itr 1 \
      --train_epochs 100 \
      --lradj 'constant' \
      --seq_len 7 \
      --pred_len 7 \
      --batch_size 4 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 3 \
      --enc_in 6 \
      --dec_in 6 \
      --c_out 6

python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/new_data6 \
      --data_path 31022519770825021X.csv \
      --model_id new_rehab_wb6 \
      --model Transformer \
      --data rehab \
      --itr 1 \
      --train_epochs 100 \
      --lradj 'constant' \
      --seq_len 7 \
      --pred_len 7 \
      --batch_size 4 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 3 \
      --enc_in 6 \
      --dec_in 6 \
      --c_out 6

python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/new_data6 \
      --data_path 31022519770825021X.csv \
      --model_id new_rehab_wb6 \
      --model DLinear \
      --data rehab \
      --itr 1 \
      --train_epochs 100 \
      --lradj 'constant' \
      --seq_len 7 \
      --pred_len 7 \
      --batch_size 4 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 3 \
      --enc_in 6 \
      --dec_in 6 \
      --c_out 6

      
      