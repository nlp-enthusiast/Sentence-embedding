#python main.py \
#  --train_data_path "./data/AllNLI.tsv"\
#  --dev_data_path "./data/stsbenchmark.tsv"\
#  --test_data_path "./data/stsbenchmark.tsv"\
#  --model_name "./plm" \
#  --save_path "./outputs" \
#  --dev --test \
#  --lr 2e-5 \
#  --num_epochs 1 \
#  --train_batch_size 128 \
#  --dev_batch_size 128 \
#  --max_seq_length 75

python main.py \
  --train_data_path "./data/queries.train.tsv" \
  --corpus_path "./data/collection.tsv" \
  --ce_scores_path "./data/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl" \
  --hard_negatives_path "./data/msmarco-hard-negatives.jsonl" \
  --dev_data_path "./data/queries.dev.tsv" \
  --test_data_path "./data/queries.test.tsv"  \
  --model_name "./plm" \
  --save_path "./outputs/text-ranking" \
  --pooling "mean" \
  --lr 2e-5 \
  --num_epochs 10 \
  --warmup_steps 1000 \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --max_seq_length 300 \
  --use_amp