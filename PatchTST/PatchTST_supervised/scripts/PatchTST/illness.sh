if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=PatchTST

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021
for pred_len in 24 36 48 60
do
    printf "22222"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'constant'\
      --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/wb_days \
      --data_path 31022519770825021X.csv \
      --model_id rehab_wb_demo \
      --model PatchTST \
      --data rehab \
      --itr 1 \
      --train_epochs 100 \
      --lradj 'constant' \
      --seq_len 7 \
      --pred_len 7 \
      --batch_size 4 \
      --patch_len 7 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 3 \
      --learning_rate 0.0001
      
python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/wb_days \
      --data_path 31022519770825021X.csv \
      --model_id rehab_wb_demo \
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
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2
      
python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/wb_days \
      --data_path 31022519770825021X.csv \
      --model_id rehab_wb_demo \
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
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2

python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/wb_days \
      --data_path 31022519770825021X.csv \
      --model_id rehab_wb_demo \
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
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2


python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/wb_days \
      --data_path 31022519770825021X.csv \
      --model_id rehab_wb_demo \
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
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2





python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/wb_days \
      --data_path 31022519770825021X.csv \
      --model_id rehab_wb_demo \
      --model PatchTST \
      --data rehab \
      --itr 1 \
      --train_epochs 1 \
      --lradj 'constant' \
      --seq_len 100 \
      --pred_len 7 \
      --batch_size 4 \
      --patch_len 7 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 3 \
      --learning_rate 0.0001 \
      --gpu 1