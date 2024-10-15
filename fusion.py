import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class MultimodalBertModel(nn.Module):
    def __init__(self, config):
        super(MultimodalBertModel, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.binary_classifier = nn.Linear(config.hidden_size, 2)
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置编码长度
    def forward(self, audio_input, text_input):
        """
        audio_input: [batch_size, seq_audio_len, hidden_size] - 已处理的音频序列嵌入
        text_input: [batch_size, seq_text_len, hidden_size] - 已处理的文本序列嵌入
        """
        device = audio_input.device
        self.bert.to(device)
        self.classifier.to(device)
        self.binary_classifier.to(device)

        # 获取批次大小
        batch_size = audio_input.size(0)

        # 创建 [CLS] 和 [SEP] token 的嵌入
        cls_token = torch.zeros((batch_size, 1, audio_input.size(-1))).to(device)
        sep_token = torch.zeros((batch_size, 1, audio_input.size(-1))).to(device)

        # 拼接音频和文本序列： [CLS] + audio_input + [SEP] + text_input + [SEP]
        audio_input = torch.cat([cls_token, audio_input, sep_token], dim=1).to(device)
        text_input = torch.cat([text_input, sep_token], dim=1).to(device)

        # 拼接后的输入序列
        inputs_embeds = torch.cat((audio_input, text_input), dim=1).to(device)

        # 检查序列长度是否超出最大限制
        if inputs_embeds.size(1) > self.max_position_embeddings:
            inputs_embeds = inputs_embeds[:, :self.max_position_embeddings]

        # 将拼接后的序列传入 BERT 模型
        outputs = self.bert(inputs_embeds=inputs_embeds)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] token 的输出

        # 分类器输出
        logits = self.classifier(cls_output)
        binary_logits = self.binary_classifier(cls_output)

        return logits, binary_logits,cls_output

