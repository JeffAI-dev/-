from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练模型和tokenizer
model_name = "gpt2"  # GPT-2模型
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 进行适配
tokenizer.pad_token = tokenizer.eos_token  # GPT-2没有pad_token，使用eos_token代替
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# 构造Dataset
train_data = Dataset.from_dict({'text': cleaned_data})

# 将文本编码为模型输入格式
def encode(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_data = train_data.map(encode, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # 训练周期
    per_device_train_batch_size=2,  # 每个设备的训练批量大小
    gradient_accumulation_steps=8,  # 梯度累积步骤
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# 使用Trainer进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# 开始训练
trainer.train()
