conda create --name SimuLineDev python=3.10 -y
conda activate SimuLineDev
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install -U scikit-learn
pip install pandas numpy matplotlib seaborn
pip install pyyaml
pip install bottleneck
pip install pydantic
pip install recbole
pip install bpemb
pip install openpyxl
pip install tqdm
pip install kmeans_pytorch

conda create --name UnbiasedEmbedding python=3.7 -y
conda activate UnbiasedEmbedding
pip install -r ./SimuLine/Preprocessing/UnbiasedEmbedding/requirements.txt
pip install tqdm
pip install protobuf==3.19.0
pip install bottleneck
pip install torch
pip install pandas numpy matplotlib seaborn

