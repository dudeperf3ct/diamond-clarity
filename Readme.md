## Diamond Clarity Classification

There are 3 classes present in the dataset. Alost equal distribution.

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

### Experiments

**CNN 3d**

- Resnet-18
- Resnet-10
- 6-layer simple cnn 3d architecture

Experiments can be tracked here : https://wandb.ai/dudeperf3ct/cnn3d

Analysis : Poor performance across all models. Performace is slightly better than random chance.
- The depth is very small (6 in our case)
- CNN 3d focus only on spatial features

-----

**CNN 3d + LSTM**

-----

**CNN 3d + GRU**