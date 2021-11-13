## Diamond Clarity Classification

There are 2 classes present in the dataset.

classes : [0, 1] => [275, 225]. The baseline accuracy to beat will be 55%. All models should perform better than this baseline.

## Installation

```bash
docker build -t dc .
./run_container.sh
```

Inside container, run

```python
python main.py
```

Inside `main.py` change `simple_cnn3d"` to either `resnet18` or `resnet10` on `line number 7`.

## Notebooks

Run the notebook on colab present in `notebooks` folder.

### Training

We use a lot of SOTA approaches like OneCycleLR, AdamW, FP16 training, Stratified splits for 90%-10% train val dataset, logging to wandb, augmentations using albumentation library. 

### Experiments

**CNN 3d**

- Resnet-18
- Resnet-10
- 6-layer simple cnn 3d architecture

Experiments can be tracked here : https://wandb.ai/dudeperf3ct/cnn3d

Analysis : 

- We overfit the model with 3 samples, only `custom_simple_cnn3d` overfits the 3 samples of each class. Other 2 model fail to overfit.


​    

-----

**CNN 3d + LSTM**

-----

**CNN 3d + GRU**