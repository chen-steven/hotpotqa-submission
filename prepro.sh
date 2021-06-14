set -e
DATA_ROOT=./data_paras/

export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/transformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

ROBERTA_LARGE=./data/models/pretrained/roberta-large

INPUT_FILE=hotpot_dev_distractor_v1.json
DATA_TYPE=hotpot_dev_distractor
OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE

[[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
#wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
#wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    
echo "1. Extract Wiki Link & NER from DB"
echo $OUTPUT_PROCESSED/doc_link_ner.json
python scripts/1_extract_db.py $INPUT_FILE data/knowledge/enwiki_ner.db $OUTPUT_PROCESSED/doc_link_ner.json

echo "2. Extract NER for Question and Context"
python scripts/2_extract_ner.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json

echo "3. Paragraph ranking"
python scripts/3_prepare_para_sel.py $INPUT_FILE $OUTPUT_PROCESSED/hotpot_ss_$DATA_TYPE.csv

python scripts/3_paragraph_ranking.py --data_dir $OUTPUT_PROCESSED --eval_ckpt data/models/finetuned/PS/pytorch_model.bin --raw_data $INPUT_FILE --input_data $OUTPUT_PROCESSED/hotpot_ss_$DATA_TYPE.csv --model_name_or_path data/models/finetuned/PS --model_type roberta --max_seq_length 256 --per_gpu_eval_batch_size 128 --fp16

echo "4. MultiHop Paragraph Selection"
python scripts/4_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED/para_ranking.json $OUTPUT_PROCESSED/multihop_para.json
