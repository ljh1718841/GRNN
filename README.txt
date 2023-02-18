Title:
GRNN: Graph retraining neural network for semi-supervised node classification
https://doi.org/10.3390/a1010000



Version:
Python 3.8.10
numpy 1.19.3
paddlepaddle-gpu   2.1.3



Running command:
Cora:
python main.py --name='cora' --hidden_size=16 --lr=0.01 --activation='elu' --l2_loss=0.0005 --feat_drop=0.5 --attn_drop=0.5 --input_drop=0.0 --num_heads 8 1 --residual=Flase --iteration_times=3 --times=1 --use_gpu=False --save_path='GRNN_data'

Citeseer:
python main.py --name='citeseer' --hidden_size=16 --lr=0.01 --activation='elu' --l2_loss=0.0005 --feat_drop=0.5 --attn_drop=0.5 --input_drop=0.0 --num_heads 8 1 --residual=Flase --iteration_times=3 --times=1 --use_gpu=False --save_path='GRNN_data'

Citeseer:
python main.py --name='pubmed' --hidden_size=16 --lr=0.01 --activation='elu' --l2_loss=0.001 --feat_drop=0.5 --attn_drop=0.0 --input_drop=0.0 --num_heads 8 8 --residual=True --iteration_times=3 --times=1 --use_gpu=False --save_path='GRNN_data'
