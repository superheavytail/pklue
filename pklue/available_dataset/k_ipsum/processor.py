"""
For VRAM-critical debugging
Now it targets 8192 context length. (longer sequence is to be updated)

IMPORTANT: This library cannot works in Windows
"""
from datasets import Dataset
import kipsum


def process(max_examples, split):
    l = []
    kip = kipsum.Kipsum()
    for i in range(1000):
        l.append({
            'chat': [
                ['user', kip.sentence(2000)],
                ['assistant', kip.sentence(2000)],
                ['user', kip.sentence(2000)],
                ['assistant', kip.sentence(2000)],
            ]
        })
    return Dataset.from_list(l)
