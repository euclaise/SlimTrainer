SlimTrainer and Adalite allow for full parameter 16-bit finetuning of language models up to 7B on a single 24GB GPU.

The optimizer uses the backpropagation fusing technique from [LOMO](https://github.com/OpenLMLab/LOMO), but uses a custom optimizer instead of using simple SGD.

The small batch size and extreme memory requirements extensive exploration of potential optimizer variants, resulting in a custom optimizer, Adalite, based on Adafactor and LAMB.

Further development is being pursued, including quantization of embedding an optimizer states.
