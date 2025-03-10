# BriLLM0.5

## Installation
```bash
pip install torch
```

## Checkpoint
[BriLLM0.5](https://huggingface.co/BriLLM/BriLLM0.5)


## Inference
```python
import json
import torch
from model import BraLM, Vocab

with open("./vocab.json") as f:
        node_dict = json.load(f)
vocab = Vocab.from_node_dict(node_dict)

model = BraLM(hidden_size=32)
model.prepare_network(vocab)

state_dict_0, state_dict_1 = torch.load("model_0.bin", weights_only=True), torch.load("model_1.bin", weights_only=True)
merged_state_dict = {**state_dict_0, **state_dict_1}
model.load_state_dict(merged_state_dict)
model.to_device("cuda:0")

head = "《罗马》描述了"
max_token = 16 - len(head)

start = [vocab((head[i]+ '->' +head[i+1])) for i in range(len(head)-1)]
ret = model.decode(start, vocab, max_token)
decode_tuple_list = [vocab.decode(p) for p in ret]
decode_sentence = decode_tuple_list[0][0] + "".join([p[-1] for p in decode_tuple_list])

print(decode_sentence)
```
