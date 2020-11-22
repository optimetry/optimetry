Optimetry
=========

[ɒpˈtɪm ɪ tri] the practice of examining optimization algorithms, by means of suitable instruments or appliances

Setup
-----
```
git clone https://github.com/optimetry/optimetry
cd optimetry
pip install -e .
```

Or, you know, just pluck from source. [https://github.com/optimetry/optimetry/blob/main/optimetry/adagraft.py].

Usage
-----
```python
# ...

from torch.optim import SGD
from your_research import CoolNewOptimizer
from optimetry import AdaGraft

M = SGD(model.parameters(), lr=3e-4)
D = CoolNewOptimizer(model.parameters())
M_graft_D = AdaGraft(M, D)  # graft M's norms onto D's directions

# ...

M_graft_D.zero_grad()
loss.backward()
M_graft_D.step()
```

Cite
----
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
- torch >= 1.7.0
