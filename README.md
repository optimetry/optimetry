optimetry
=========

[ɒpˈtɪm ɪ tri] the practice of examining optimization algorithms, by means of suitable instruments or appliances

Setup
-----
```
git clone https://github.com/optimetry/optimetry
cd optimetry
pip install -e .
```

Or, you know, just [pluck from source](https://raw.githubusercontent.com/optimetry/optimetry/main/optimetry/graft.py).

Usage
-----
```python
# ...

from torch.optim import SGD
from your_research import CoolNewOptimizer
from optimetry import Graft

M = SGD(model.parameters(), lr=3e-4)
D = CoolNewOptimizer(model.parameters())
MxD = Graft(M, D)  # graft M's norms onto D's directions

# ...

MxD.zero_grad()
loss.backward()
MxD.step()
```

Cite
----
[Why?](https://arxiv.org/abs/2002.11803)
```
@article{agarwal2020disentangling,
  title={Disentangling Adaptive Gradient Methods from Learning Rates},
  author={Agarwal, Naman and Anil, Rohan and Hazan, Elad and Koren, Tomer and Zhang, Cyril},
  journal={arXiv preprint arXiv:2002.11803},
  year={2020}
}
```

Requirements
------------
- Python >= 3.6
- torch >= 1.7.0

See also
--------
- [TF 1.x code](https://tensorflow.github.io/lingvo/_modules/lingvo/core/adagraft.html#AdaGraftOptimizer) lives within Lingvo.
