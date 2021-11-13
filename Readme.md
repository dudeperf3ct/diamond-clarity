## Diamond Clarity Classification

There are 3 classes present in the dataset. Not equal distribution.

Classes : [0, 1] -> [275, 225]. The baseline accuracy of model will be 55%.

## Installation

```bash
docker build -t dc .
./run_container.sh
```

Inside container, run

```python
python main.py
```

Since insufficient memory, run the notebook in `notebook` folder on colab.

### Training

This version uses tensorflow to train the dataset.

### Experiments

**CNN 3d**

- Simple CNN3d

Experiments can be tracked here : https://wandb.ai/dudeperf3ct/cnn3d_tf



-----

**CNN 3d + LSTM**

-----

**CNN 3d + GRU**