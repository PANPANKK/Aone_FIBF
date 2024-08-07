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

    def forward(self, audio_input, text_input, attention_mask, token_type_ids):
        device = audio_input.device
        self.bert.to(device)
        self.classifier.to(device)
        self.binary_classifier.to(device)

        text_input = text_input.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        batch_size = audio_input.size(0)
        cls_token = torch.zeros((batch_size, 1, audio_input.size(-1))).to(device)
        sep_token = torch.zeros((batch_size, 1, audio_input.size(-1))).to(device)

        audio_input = torch.cat([cls_token, audio_input, sep_token], dim=1).to(device)
        text_input = torch.cat([text_input, sep_token], dim=1).to(device)

        inputs_embeds = torch.cat((audio_input, text_input), dim=1).to(device)

        # 检查输入序列长度
        if inputs_embeds.size(1) > self.max_position_embeddings:
            inputs_embeds = inputs_embeds[:, :self.max_position_embeddings]
            attention_mask = attention_mask[:, :self.max_position_embeddings]
            token_type_ids = token_type_ids[:, :self.max_position_embeddings]

        # Ensure all tensors are on the same device
        attention_mask = attention_mask.to(inputs_embeds.device)
        token_type_ids = token_type_ids.to(inputs_embeds.device)

        # Pass through BERT model
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        binary_logits = self.binary_classifier(cls_output)

        return logits, binary_logits


def create_attention_mask_and_token_type_ids(audio_seq_input, audio_attention_masks, text_seq_input, max_length):
    device = audio_seq_input.device  # 获取audio_seq_input的设备
    batch_size, seq_audio_len, _ = audio_seq_input.shape
    _, seq_text_len, _ = text_seq_input.shape

    # Ensure the total length does not exceed max_length
    total_length = seq_audio_len + seq_text_len + 3  # [CLS] + audio + [SEP] + text + [SEP]
    if total_length > max_length:
        # Calculate the number of tokens to truncate
        truncate_len = total_length - max_length
        if seq_text_len > truncate_len:
            text_seq_input = text_seq_input[:, :-truncate_len, :].to(device)
            seq_text_len -= truncate_len
        else:
            audio_seq_input = audio_seq_input[:, :-(truncate_len - seq_text_len), :].to(device)
            audio_attention_masks = audio_attention_masks[:, :-(truncate_len - seq_text_len)].to(device)
            seq_audio_len -= (truncate_len - seq_text_len)

    # Create text attention mask (assuming no padding for text inputs)
    text_attention_masks = torch.ones((batch_size, seq_text_len), dtype=torch.long).to(device)

    # Create attention mask
    attention_mask = torch.cat((
        torch.ones((batch_size, 1), dtype=torch.long).to(device),  # CLS token
        audio_attention_masks.to(device),
        torch.ones((batch_size, 1), dtype=torch.long).to(device),  # SEP token after audio
        text_attention_masks.to(device),
        torch.ones((batch_size, 1), dtype=torch.long).to(device)  # SEP token after text
    ), dim=1)

    # Create token type ids
    token_type_ids = torch.cat((
        torch.zeros((batch_size, 1 + seq_audio_len + 1), dtype=torch.long).to(device),  # CLS + audio + SEP tokens
        torch.ones((batch_size, seq_text_len + 1), dtype=torch.long).to(device)  # text + SEP tokens
    ), dim=1)

    return attention_mask, token_type_ids


# 增加一个创建二分类样本对的函数
def create_binary_classification_pairs(audio_embeddings, audio_attention_masks, text_embeddings, labels):
    pairs = []
    pairs_audio_mask = []
    pair_labels = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                pairs.append((audio_embeddings[i], text_embeddings[j]))
                pairs_audio_mask.append(audio_attention_masks[i])
                pair_labels.append(1 if labels[i] == labels[j] else 0)
    return pairs, pairs_audio_mask, pair_labels


# Define the model configuration
config = BertConfig(
    hidden_size=1024,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=4096,
    max_position_embeddings=2800,
    num_labels=4
)


###...
def create_attention_mask_and_token_type_ids_for_binary(binary_pairs_audio, binary_pairs_masks, binary_pairs_text,
                                                        max_length):
    batch_size = binary_pairs_audio.size(0)
    seq_audio_len = binary_pairs_audio.size(1)
    seq_text_len = binary_pairs_text.size(1)

    # Ensure the total length does not exceed max_length
    total_length = seq_audio_len + seq_text_len + 3  # [CLS] + audio + [SEP] + text + [SEP]

    if total_length > max_length:
        # Calculate the number of tokens to truncate
        truncate_len = total_length - max_length
        if seq_text_len > truncate_len:
            binary_pairs_text = binary_pairs_text[:, :-truncate_len, :]
            seq_text_len -= truncate_len
        else:
            binary_pairs_audio = binary_pairs_audio[:, :-(truncate_len - seq_text_len), :]
            seq_audio_len -= (truncate_len - seq_text_len)

    # Create attention mask for binary pairs
    attention_mask = torch.cat((
        torch.ones((batch_size, 1), dtype=torch.long).to(binary_pairs_audio.device),  # CLS token
        binary_pairs_masks.to(binary_pairs_audio.device),  # audio padded token masks
        torch.ones((batch_size, 1), dtype=torch.long).to(binary_pairs_audio.device),  # SEP token after audio
        torch.ones((batch_size, seq_text_len), dtype=torch.long).to(binary_pairs_audio.device),  # text tokens
        torch.ones((batch_size, 1), dtype=torch.long).to(binary_pairs_audio.device)  # SEP token after text
    ), dim=1)

    # Create token type ids for binary pairs
    token_type_ids = torch.cat((
        torch.zeros((batch_size, 1 + seq_audio_len + 1), dtype=torch.long).to(binary_pairs_audio.device),
        # CLS + audio + SEP tokens
        torch.ones((batch_size, seq_text_len + 1), dtype=torch.long).to(binary_pairs_audio.device)  # text + SEP tokens
    ), dim=1)

    return attention_mask, token_type_ids