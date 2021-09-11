# AMP prediction  
## Using heterogeneous CNN (ResNet, DenseNet, SENet) to predict AMP (Antimicrobial Peptide).  

## Directory Structure  
- `data/`: protein sequences  
    - benchmark/: Xiao dataset from paper `iAMP-2L`  
    - scanner/: Veltri dataset from paper `AMPScanner`  
- `util/`: functions for training, jupyter notebooks for visualization    
- `results/`: save models and log  
    - benchmark/
        - SENet/  
        - SENet_nopool/  
        - ConcatNet/  
        - ScannerNet/  
    - scanner/  
        - SENet/  
        - SENet_nopool/  
        - ConcatNet/  
        - ScannerNet/  

## Experiment  

### Prepare data  
positive sequences: APD database  
negative sequences: Uniprot  
protein sequences in `data`:  
- `benchmark/fa/`: five categories(bacterial, cancer, fungal, viral, hiv) in positive data 
    - `tr_pos.fasta`  
    - `tr_neg.fasta`  
    - `te_pos.fasta`  
    - `te_neg.fasta`  
    - `tr_bacterial.fasta`  
    - `te_bacterial.fasta`  
    - `tr_cancer.fasta`  
    - `te_cancer.fasta`  
    - `tr_fungal.fasta`  
    - `te_fungal.fasta`  
    - `tr_viral.fasta`  
    - `te_viral.fasta`  
    - `tr_hiv.fasta`  
    - `te_hiv.fasta`  
- `scanner/fa/`:  
    - `tr_pos.fasta`  
    - `tr_neg.fasta`  
    - `va_pos.fasta`  
    - `va_neg.fasta`  
    - `te_pos.fasta`  
    - `te_neg.fasta`  

To encode the amino acids:  
```
cd data/
python3 preprocess.py
```
After `preprocess.py`, encoded data will be generated in `benchmark/seq/` and `scanner/seq/`.  
Then we use these pkl files to train models.  

### Train models
```
python3 train.py --data [dataset] --model [model_structure]
```
- dataset: `benchmark`/`scanner`  
- model_structure: model structures in `util/net.py`, `SENet`/`SENet_nopool`/`ConcatNet`/`ScannerNet`  
    - SENet: 4 layer Conv with SE block -> used to visualize 4 SE blocks  
    - SENet_nopool: 4 layer Conv with SE block, without avgpooling -> used to visualize amino acid importance  
    - ConcatNet: the best model in this experiment
    - ScannerNet: the structure in paper `AMPScanner`

### Results  
Each experiment will run 20 times, all results will be saved in `results/`  
- `results/benchmark/SENet`  
- `results/benchmark/SENet_nopool`  
- `results/benchmark/ConcatNet`  
- `results/benchmark/ScannerNet`  
- `results/scanner/SENet`  
- `results/scanner/SENet_nopool`  
- `results/scanner/ConcatNet`  
- `results/scanner/ScannerNet`  
  
In each folder, there are 20 model files: `model_0.pkl`, `model_1.pkl`, ......, `model_19.pkl`.  
The average of these 20 models is recorded in `results.log`.

### Visualize
`util/plot_SElayer.ipynb`: visualize 4 SE blocks, using models in `results/benchmark/SENet/`.  
`util/visualize.ipynb`: visualize where model focus on sequence, using models in `results/benchmark/SENet_nopool`.  
