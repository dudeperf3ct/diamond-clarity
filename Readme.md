## Diamond Clarity Classification

There are 3 classes present in the dataset. Not equal distribution.

Classes : [0, 0.5, 1] -> [233,  42, 225]

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