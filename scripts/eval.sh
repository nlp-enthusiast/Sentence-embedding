python eval.py \
  --corpus_path "./data/collection.tsv" \
  --ce_scores_path "./data/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl" \
  --hard_negatives_path "./data/msmarco-hard-negatives.jsonl" \
  --test_data_path "./data/queries.eval.tsv"  \
  --model_name "./outputs/text-ranking" \
  --dev_batch_size 32