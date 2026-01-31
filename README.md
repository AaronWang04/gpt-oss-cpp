Attempt at lightweight, standalone c++ inference engine for gpt-oss

download the model
```
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b-model/
hf download openai/gpt-oss-20b --include "*config" --local-dir gpt-oss-20b-model/
```

download the tokenizer
```
wget -O gpt-oss-20b-model/o200k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
```

you may need to download the regex (cuz std::regex doesn't support unicode)
```
apt-get install -y libicu-dev
```

standard stuff for cmake projects
initialize the configure dir
```
mkdir build && cd build && cmake ..
```
then build the project and run it
```
cmake --build build && ./build/gptoss
```
