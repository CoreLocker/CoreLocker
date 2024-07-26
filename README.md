## CoreLocker: Neuron-level Usage Control
### This is a demo implementation for the paper [CoreLocker: Neuron-level Usage Control]

**Project Structure**

```
CoreLocker
├── README.md
├── corelock.py
├── extractor.py
└── models
    ├── __init__.py
    ├── channel_selection.py
    ├── densenet.py
    └── preresnet.py
```

**To split the target model:**

```python corelock.py split --model <path_to_model> --ratio <key_ratio> --arch <architecture> --save ./test_output```

**To patch the model:**

```python corelock.py recover --model test_output/keyed_model.pth.tar --key test_output/key.pth.tar --arch <architecture> --save <save_path>```
