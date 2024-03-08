---
title: "ONNX Tutorial"
date: 2021-10-21T15:34:30+09:00
categories:
  - machine learning
tags:
  - machine learning
  - deep learning
  - onnx
  - pytorch
  - tensorflow
  - huggingface
---

- 본 튜토리얼은 pytorch와 huggingface model에 초점이 맞추어져 있음을 알립니다.

## ONNX 개요

- ONNX [ˈo:nʏks] - Open Neural Network eXchange
- 머신러닝 모델을 표현하는 오픈 표준 포맷
- 제공 기능
  - Build model / Export to ONNX format
    - **Tensorflow, Pytorch, Scikit Learn** 등 20여 가지의 Framework & Converter 지원
    - Pre-Trained model (**ONNX Model Zoo**) 제공
  - Deploy model
    - Intel, Qualcomm, Windows, Nvidia, TensorFlow 등의 다양한 플랫폼 지원
  - [ONNX Optimizer](https://github.com/onnx/optimizer)
  - Model Visualize - [Netron](https://github.com/lutzroeder/Netron)

## ONNX Runtime

- 머신러닝의 추론 및 학습을 최적화하는 기능

## 사용 준비

- prerequisite : pytorch (or tensorflow)
- Install ONNX and ONNX runtime.

```sh
conda install -c conda-forge onnx # no need when using pytorch.
pip install onnxruntime # when using CPU
pip install onnxruntime-gpu # when using GPU
```

## ONNX 변환 튜토리얼 (PyTorch model)

### 개요

HuggingFace에서 제공하는 PyTorch 기반 BERT 모델을 ONNX 포맷으로 변환한다.

### transformers.onnx 을 사용한 변환

- hub 또는 local path에 있는 모델을 ONNX graph로 변환한다.
- 예시: bert-base-cased 모델을 onnx/bert-base-cased/ 위치에 변환

```sh
python -m transformers.onnx --model=bert-base-cased onnx/bert-base-cased/
```

- `transformers.onnx` 의 매개변수 목록

```txt
usage: Hugging Face ONNX Exporter tool [-h] -m MODEL -f {pytorch} [--features {default}] [--opset OPSET] [--atol ATOL] output

positional arguments:
  output                Path indicating where to store generated ONNX model.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model's name of path on disk to load.
  --features {default}  Export the model with some additional features.
  --opset OPSET         ONNX opset version to export the model with (default 12).
  --atol ATOL           Absolute difference tolerance when validating the model.
```

### Python API 를 사용한 변환

```py
import torch

torch.onnx.export(model,
                  dummy_input,
                  "onnx/bert-base-cased/model.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=dynamic_axes)
```

#### 매개변수 설명

- model: export할 pytorch model
- dummy_input: model에 들어가는 단일 input 또는 튜플 형태의 다중 input
  - **다중 input 이라면 튜플 속에 PreTrainedModel.forward()에 들어가는 인수의 순서대로 넣어주어야 함**
- 모델을 export할 경로 (type : str)
- input_names: nput으로 들어가는 변수의 이름들 (type : list[str])
- output_names: output으로 나오는 변수의 이름들 (type : list[str])
- dynamic_axes: 가변 길이의 input/output 차원이 있을 때 알려주는 변수 (type : dict[str, dict[int, str]])

### ONNX 모델을 ONNX Runtime에서 사용하기

```py
import onnxruntime as ort

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

ort_session = ort.InferenceSession("onnx/bert-base-cased/model.onnx")

inputs = tokenizer("Using BERT in ONNX!", return_tensors="np")
outputs = ort_session.run(["last_hidden_state", "pooler_output"], dict(inputs))
```

`ort_session.run`에 들어가는 output key들은 모델의 onnx config를 통해 확인하거나, 직접 input을 model에 넣어서 어떤 값이 나오는지 확인하면 된다. 👇

```py
from transformers.models.bert import BertOnnxConfig, BertConfig

config = BertConfig()
onnx_config = BertOnnxConfig(config)
output_keys = list(onnx_config.outputs.keys())
```

### 지원되지 않는 아키텍쳐에 대한 custom config

- custom config를 정의할 수 있다.
- 단 `inputs`, `outputs` property는 `OrderedDict`이므로 순서를 맞추는 것이 중요하다.
  - `inputs`는 `PreTrainedModel.forward()`의 프로토타입과,<br>`outputs`는 `BaseModelOutputX` 인스턴스와 매개변수 위치가 맞아야 한다.

```py
class CustomOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([
          ("input_ids", {0: "batch", 1: "sequence"}),
          ("attention_mask", {0: "batch", 1: "sequence"}),
          ("token_type_ids", {0: "batch", 1: "sequence"}),
        ])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([
          ("last_hidden_state", {0: "batch", 1: "sequence"}),
          ("pooler_output", {0: "batch"})
        ])
```

### PyTorch → ONNX 변환 예시 코드

<https://github.com/lhy0718/huggingface-study/blob/main/onnx.ipynb>

## references

> - <https://huggingface.co/transformers/serialization.html>
