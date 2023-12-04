# Learning Path

## Building a GPT from scratch
[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

[Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/): This course is done by OpenAI and teaches you how to fine-tune OpenAI models.

## 3 ways to do fine tuning
According to a [medium article](https://medium.com/p/23473d763b91),written by Shawhin Talebi, there are in general three ways of **fine-tuning** LLMs:
### Self-supervised Learning
### Supervised Learning
### Reinforcement Learning
You can find different fine-tuning tutorials at this [link](https://github.com/ashishpatel26/LLM-Finetuning). The tutorials here are indeed comprehensive, they range from OpenAI models to the HuggingFace, SFT, TRL, Reward modelling, and PPO.
# LLM-and-FineTuning
 One good source for interview questions would be [Master Your ML & DS Interview](https://www.mlstack.cafe/blog/large-language-models-llms-interview-questions)

A useful link, explaining all the topics below can be found at [In-depth guide to fine-tuning LLMs with LoRA and QLoRA](https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora#:~:text=QLoRA%20and%20LoRA%20both%20are,of%20a%20standalone%20finetuning%20technique.)

## Transformer documentation on HuggingFace
## Distributed training with 🤗 Accelerate
## Adapter Layers
### Causal LLM’s, Masked LLM’s, and Seq2Seq
[Causal LLM’s, Masked LLM’s, and Seq2Seq](https://medium.com/@tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa)
## PEFT
Check out the [PEFT](https://github.com/huggingface/peft) repo on the GitHub.
## how to leverage RL in finetuning
## TRL
## Reinforcement Learning from Human Feedback (RLHF)
## LORA
the math behind LORA
## LORA + Int 8bit quantization
## QLORA
## Catastrophic forgetting
## what is LAMINI
Lamini is the LLM platform for enterprises and developers to build customized, private models: easier, faster, and higher-performing than any general LLMs.


## Parallelization
In case you are interested in knowing about parallelization, follow the [link](https://towardsdatascience.com/how-to-build-an-llm-from-scratch-8c477768f1f9).




# Steps to fine-tune an LLM
## choose a fine tuning task
### text summarization
### text classification

## Prepare training dataset
### pre-process the data
#### tokenize
#### choose DataCollatorWithPadding
## Choose a base model
### you can use bits-and-bytes to load quantized models

```
 bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype= torch.float16,
     bnb_4bit_use_double_quant=False,
```



### using lora_config to using parameter efficient fine-tuning
(we freeze all the parameters, but we augment the model with additional parameters that are trainable)


```
peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules=["query"],)
```
* r: the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
* target_modules: The modules (for example, attention blocks) to apply the LoRA update matrices.
* alpha: LoRA scaling factor.
* bias: Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'.
* modules_to_save: List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task.

## Fine Tune a model via supervised learning 
## Evaluate the performance
### run inference
