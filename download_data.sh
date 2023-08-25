data_path="./data"
mkdir -p ${data_path}
cd ${data_path}
wget https://sbert.net/datasets/AllNLI.tsv.gz
wget https://sbert.net/datasets/stsbenchmark.tsv.gz
gzip -d AllNLI.tsv.gz
gzip -d stsbenchmark.tsv.gz
cd ..