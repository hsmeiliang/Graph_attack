# Data Science HW4

## Please install anaconda & cuda 11.3 (if you want to use gpu) first.

## Create the Environment
```
conda create --name hw4 -y
conda activate hw4
conda install scipy numpy -y
```
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
```
* without gpu support
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

## Python version
```
Python 3.8.8
```

## Package version
```
numba 0.53.1
numpy 1.20.1
torch 1.11.0
scipy 1.6.2
```

## Run
```
python3 main.py 
```
* You can pass arguments like this.
```
python3 main.py --input_file target_nodes_list.txt --data_path ./data/data.pkl --model_path saved-models/gcn.pt
```

## Dataset
* Cora citation network
* Select the largest connected components of the graph and use 10%/10% nodes for training/validation.
* Stats:
  
| #nodes | #edges | #features | #classes |
|--------|--------|-----------|----------|
| 2485   | 10138  | 1433      | 7        |

## TODO
* attacker.py
  * implement your own attacker
* main.py
  * setup your attacker
