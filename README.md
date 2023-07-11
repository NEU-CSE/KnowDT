# KnowDT
> The open-source code for the paper *KnowDT: Empathetic Dialogue Generation with Knowledge Enhanced Dependency Tree*.


## Dependencies

Install the required libraries (Python 3.8.11 | CUDA 10.2)

```sh
pip install -r requirements.txt 
```

Download  [**Pretrained GloVe Embeddings**](http://nlp.stanford.edu/data/glove.6B.zip) and save it in `/vectors`.

## Dataset
The file size is out of limit, if necessary, please email: 2539719155@qq.com for pre-processed data.

## Training

```sh
python main.py --model KnowDT
```

The hyperparameters and ablation experiments can be modified in `/src/utils/config.py`.

## Testing
Make sure the folder `save/test` is not empty and run the following:

```sh
python main.py --model KnowDT --test --model_path save/test/[model-name]
```

## Evaluation

Move the obtained results.txt to the folder `results` and run the following:

```sh
python src/scripts/evaluate.py #PPL, Dist 1/2, Acc
python Metrics/EmpatheticMetric.py #Rouge, BLEU
```

