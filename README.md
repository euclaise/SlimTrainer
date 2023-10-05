This repository is currently in an early, expiremental stage, and should not be considered production-ready.

SlimTrainer and Adalite allow for full parameter 16-bit finetuning of language models up to 7B on a single 24GB GPU.

The optimizer uses the backpropagation fusing technique from [LOMO](https://github.com/OpenLMLab/LOMO), but uses a custom optimizer instead of using simple SGD.

The small batch size and extreme memory requirements extensive exploration of potential optimizer variants, resulting in a custom optimizer, Adalite, based on Adafactor and LAMB.

Further development is being pursued, including quantization of embedding and optimizer states, which should allow for larger batch sizes.

Note that long sequence lengths require more memory. A sequence length of 512 at a batch size of 1, with flash attention and Adalite, for LLaMA 7B, just fits at a batch size of 1 (and a batch size of 2 is so close that I suspect it could be achieved with some efficiency tweaks). StableLM 3B fits at a batch size of 4.

An implementation of [Sigma reparameterization](https://proceedings.mlr.press/v202/zhai23a/zhai23a.pdf) and a naive version of embedding quantization are also present in this repository. Apple found that sigma parameterization was sufficient to allow them to pretrain a transformer with SGD+momentum. Unfortunately, momentum alone takes too much memory, and I have not seen any improvement for finetuning using sigma reparameterization with non-momentum SGD.

The techniques used in this repository should also help with training in less constrained settings, but I haven't tested it.  In such contexts, with a sufficiently large batch size, the `OverlapLion` optimizer may be useable.
