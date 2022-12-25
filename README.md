Currently there are some problem managing pytorch using poetry. A temporary solution is to run the following to install torch manually.

```
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```
