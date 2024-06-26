<h1>DUnet</h1>
A 3D convolutional neural network DUnet that derived from DenseNet and UNet for predicting the protein-ligand binding sites.

The datasets can be downloaded from:
https://github.com/devalab/DeepPocket
https://github.com/jivankandel/PUResNet
Please cite the indicated references while using the given dataset.

<h1>Environment Preparetion</h1>
1. clone this repository
<pre>
git clone https://github.com/wangxinglong1990/DUnet.git
cd DUnet
</pre>
2. install Anaconda under the in instruction in https://www.anaconda.com/  
Then run the following code to create the environment:
<pre>
conda env create -f environment.yml
</pre>
3. intstall tfbio from https://gitlab.com/cheminfIBB/tfbio

<h1>Usage</h1>
We provide two method for protein-ligand area segmentation, including DenseNet and self-attention-based (DUnet), and DenseNet-based model.

</pre>
To train the model with other database, 2 step shall be followed:<br>
1. generate the dataset from database, run the following command (data preprocessing may need to be done if using your own database):  
<pre>
python generate_dataset.py --database [path of database]
</pre>
example:  
<pre>
python generate_dataset.py --database scPDB_original/
</pre>
Then a fold named "dataset" will be generate with "taining_set.npy" and "test_set.npy" in it, which is required by the training program.<br>
2. train the model, using the following command:
<pre>
python train.py --dataset [path of dataset] --bz [batch size] -- lr [learning rate] --epoch [number of epochs]
</pre>
example:
<pre>
python train.py --dataset dataset/ --bz 5 -- lr 1e-4 --epoch 100
</pre>  

AlphaFold structures predicted binding sites:  
https://drive.google.com/file/d/1hev9l8OiVf7Cti7VIfj7kce0bYuEFkDW/view?usp=sharing  




<h1>Citation</h1>
DUnet: A deep learning guided protein-ligand binding pocket prediction

Xinglong Wang, Beichen Zhao, Penghui Yang, Yameng Tan, Ruyi Ma, Shengqi Rao, Jianhui Du, Jian Chen, Jingwen Zhou, Song Liu

bioRxiv 2022.08.11.503579; doi: https://doi.org/10.1101/2022.08.11.503579
