# Laboratory work №2

1. Run compression with the following command:
```
python compress.py --path-to-model "Qwen/Qwen3-8B" --path-to-save "data/Qwen3-8B-8bit"
```

2. Evaluate compression ratio, performance drop, and calculate score with the following command:
```
python inference.py --path-to-original-model "Qwen/Qwen3-8B" --path-to-compressed-model "data/Qwen3-8B-8bit" --batch-size "10"
```

> Compressed weights are located on [Google Drive](https://disk.yandex.ru/d/YF2ulDK1XJKKTQ/Qwen3-8_int8), put them into `data/Qwen3-8B-8bit` folder

**Obtained results**

| Criteria          |  Result |
|:-----------------:|:-------:|
| Compression Ratio |  1.74   |
| Performance Drop  |  0.0021 |
| Score             |  1.73   |

# Laboratory work №3

1. Run LoRA fine-tuning with the following command:
```
python train.py --model "data/Qwen3-8B-8bit"
```

2. Evaluate compression ratio, performance drop, and calculate score with the following command:
```
python inference.py --path-to-original-model "Qwen/Qwen3-8B" --path-to-compressed-model "data/Qwen3-8B-8bit" --peft-model "data/lora" --batch-size "10"
```

> Fine-tuned weights are located on [Google Drive](https://disk.yandex.ru/d/YF2ulDK1XJKKTQ/Qwen3-8B_int8_fine_tuned), put them into `data/lora` folder

**Obtained results**

| Criteria          |  Result |
|:-----------------:|:-------:|
| Compression Ratio |  1.72   |
| Performance Drop  |  0.0009 |
| Score             |  1.72   |
