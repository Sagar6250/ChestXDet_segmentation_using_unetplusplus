# Dataset

ChestX-Det is publicly available on [Deepwise-AILab github](https://github.com/Deepwise-AILab/ChestX-Det-Dataset) or [Kaggle](https://www.kaggle.com/datasets/mathurinache/chestxdetdataset)

# Requirements

The model used is a Unet++ implementation from [Segmentation model pytorch](https://github.com/qubvel/segmentation_models.pytorch). Install it using

```
$ pip install -U segmentation-models-pytorch
```

# Running

To train the model, run

```
python main.py --experiment experiment-name --resume Path(if needed)
```

To evaluate the model, run

```
python main.py --eval Path
```

To get inference masks, run

```
python infer.py
```
