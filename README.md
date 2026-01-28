Attempt at lightweight, standalone c++ inference engine for gpt-oss

download the model
```
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b-model/
```

download the tokenizer
```
wget -O gpt-oss-20b-model/o200k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
```
