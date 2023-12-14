import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import os
os.environ['CURL_CA_BUNDLE'] = ''


# 数据集准备
class ContractDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.max_length = max_length
        self.data, self.labels = self.load_data(data_path)
        self.tokenizer = tokenizer

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        data = [line.strip() for line in lines]
        labels = [1 if "漏洞" in line else 0 for line in lines]  # 假设"漏洞"为异常标签
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        contract_code = self.data[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(contract_code, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=self.max_length)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': label
        }


# 创建BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 数据集文件路径
# data_path = 'contract_data.txt'
data_path = 'D:\\project\\smartbugs-bte\\contracts_test1.txt'


# 创建数据集实例
dataset = ContractDataset(data_path, tokenizer)

# 划分训练集和验证集
train_size = 0.8
train_dataset, val_dataset = train_test_split(dataset, train_size=train_size, random_state=42)

# 创建数据加载器
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 构建模型
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out


# 创建模型实例
embedding_dim = 128
hidden_dim = 64
num_layers = 2
num_classes = 2  # 2个类别：正常和异常
vocab_size = len(tokenizer)

bilstm_model = BiLSTMModel(embedding_dim, hidden_dim, num_layers, num_classes)


# 解释性Transformer模型
class ExplainerTransformer(nn.Module):
    def __init__(self):
        super(ExplainerTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased',resume_download=True)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 取CLS token的输出作为特征
        logits = self.fc(pooled_output)
        return logits


explainer_model = ExplainerTransformer()

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(bilstm_model.parameters()) + list(explainer_model.parameters()), lr=0.001)

# 训练循环
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bilstm_model.to(device)
explainer_model.to(device)

for epoch in range(num_epochs):
    bilstm_model.train()
    explainer_model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # 使用Bi-LSTM提取每行代码的语义特征
        lstm_outputs = bilstm_model(input_ids, attention_mask)

        # 提取全局特征（采用平均池化）
        global_features = lstm_outputs.mean(dim=1)  # 对每个时间步进行平均池化

        # 结合语义特征和全局特征，使用解释性Transformer进行预测
        combined_features = torch.cat((lstm_outputs, global_features.unsqueeze(1).expand(-1, lstm_outputs.size(1), -1)),
                                      dim=2)
        combined_features = combined_features.permute(0, 2, 1)  # 调整维度顺序以匹配transformer的输入
        transformer_logits = explainer_model(combined_features, attention_mask)

        loss = criterion(transformer_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
