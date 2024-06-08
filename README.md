# CodeGemma 2B Model Fine-Tuning and Inference

## Installation Instructions

1. Install the required libraries:
    ```bash
    !pip install -q accelerate transformers peft bitsandbytes trl
    ```

2. Login to Hugging Face to access the Gemma 2B models:
    ```bash
    !huggingface-cli login
    ```
    Generate a token from [Hugging Face](https://huggingface.co/settings/tokens) and enter it when prompted.

## Training Instructions

1. Initialize the model and tokenizer:
    ```python
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

    model_id = "google/codegemma-2b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = GemmaTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
    ```

2. Prepare the model for training:
    ```python
    from peft import prepare_model_for_kbit_training

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    ```

3. Print trainable parameters:
    ```python
    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    print_trainable_parameters(model)
    ```

4. Identify linear layers:
    ```python
    import bitsandbytes as bnb

    def find_all_linear_names(model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) is 1 else names[-1])
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
        return list(lora_module_names)

    modules = find_all_linear_names(model)
    print(modules)
    ```

5. Configure LoRA parameters:
    ```python
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    ```

6. Load and preprocess the dataset:
    ```python
    from datasets import load_dataset

    data = load_dataset("your_dataset_name", split="train").select(range(7000))
    data = data.map(lambda samples: tokenizer(samples["code"]), batched=True)
    ```
    Replace "your_dataset_name" with the name of the dataset you want to use. Ensure that the dataset has a column named "code" containing Python code snippets. If the dataset is not available in the Hugging Face library, you can upload your dataset or provide a link to it in a compatible format such as CSV or JSON. Then load it using load_dataset() function by specifying the appropriate name and split.



7. Train the model:
    ```python
    import transformers

    tokenizer.pad_token = tokenizer.eos_token
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    ```
8. Saving the Fine Tuned Model
   ```python
     model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
     model_to_save.save_pretrained("outputs")
   ```
9. Loading the finetuned model
    ```python
      lora_config = LoraConfig.from_pretrained('outputs')
      model = get_peft_model(model, lora_config)
    ```
10. Generating inference of the model
    ```python
      text = "def add_two_numbers(): "
      device = "cuda:0"
      
      inputs = tokenizer(text, return_tensors="pt").to(device)
      outputs = model.generate(**inputs, max_new_tokens=100)
      print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```
