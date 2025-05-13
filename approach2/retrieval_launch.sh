file_path=/home/s3648885/data0/Search-R1/downloaded_data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=intfloat/e5-base-v2

python /home/s3648885/data0/Search-R1/mcts_src/approach2/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever > ~/data0/Search-R1/retrieval_server.log 2>&1 &