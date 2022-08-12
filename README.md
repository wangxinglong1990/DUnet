<h1>DUnet (The codes and detailed information will be updated when the preprint is published)</h1>

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
To predict the protein pocket with DUnet, run the following command:
<pre>
python predict.py --protein [path of pdb file] --model [saved model]
</pre>
example:
<pre>
python predict.py --protein 1iu4.pdb --model DUnet-3.pth
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
python train.py --dataset dataset/ --bz 1 -- lr 1e-4 --epoch 200
</pre>

<h1>Citation</h1>
