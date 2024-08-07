import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torchaudio
import numpy as np
from iemocap_dataloader import *
from multi_modality_fusion import *
from tqdm import tqdm
import random
import torch.nn.functional as F
import os
import copy  # 导入deepcopy


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# 定义投影头
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# 定义带有可学习温度系数的模型
class TemperatureModel(nn.Module):
    def __init__(self, initial_temp=1.0):
        super(TemperatureModel, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temp))

    def forward(self):
        return torch.sigmoid(self.temperature)  # 将温度系数限制在0到1之间


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temperature_model_init = TemperatureModel().to(device)

# RoBERTa 配置
roberta_tokenizer = RobertaTokenizer.from_pretrained('../roberta-large-uncased')
roberta_model_init = RobertaModel.from_pretrained('../roberta-large-uncased').to(device)


# 获取文本向量全局和序列向量
class Get_TextEmbedding(nn.Module):
    def __init__(self, roberta_tokenizer, text_model):
        super(Get_TextEmbedding, self).__init__()
        self.tokenizer = roberta_tokenizer
        self.text_model = text_model

    def forward(self, text_inputs):
        text_embeddings = self.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True,
                                         max_length=80).to(device)
        text_output = self.text_model(**text_embeddings)
        text_seq_embedding, text_cls_embeddings = text_output[0], text_output[1]
        return text_seq_embedding, text_cls_embeddings


# 创建情感预测模型
num_labels = 4
hidden_size = roberta_model_init.config.hidden_size
TextEmbedding_model_init = Get_TextEmbedding(roberta_tokenizer, roberta_model_init).to(device)

bert_config = BertConfig(
    hidden_size=1024,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=4096,
    max_position_embeddings=2800,
    num_labels=4
)

Bert_adapter_multimodel_fusion_init = MultimodalBertModel(bert_config)
projection_head_init = ProjectionHead(input_dim=1024, output_dim=1024).to(device)

classification_criterion = nn.CrossEntropyLoss()


def evaluate_model(model1, model2, dataloader):
    model1.eval()
    model2.eval()
    temperature_model_init.eval()
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0
    class_correct = np.zeros(num_labels)
    class_total = np.zeros(num_labels)
    with torch.no_grad():
        for batch in dataloader:
            raw_audio_seq_input_padded, aug_audio_seq_input_padded, raw_audio_cls_input, aug_audio_cls_input, \
            raw_audio_attention_masks, aug_audio_attention_masks, raw_audio_dimension, aug_audio_dimension, \
            raw_text, aug_text, labels = batch['raw_audio_seq_input_padded'], batch['aug_audio_seq_input_padded'], \
                                         batch['raw_audio_cls_input'], \
                                         batch['aug_audio_cls_input'], batch['raw_audio_attention_masks'], batch[
                                             'aug_audio_attention_masks'], \
                                         batch['raw_audio_dimension'], batch['aug_audio_dimension'], batch['raw_text'], \
                                         batch['aug_text'], batch['label']

            raw_audio_cls_embeddings = raw_audio_cls_input.to(device).squeeze(1)
            aug_audio_cls_embeddings = aug_audio_cls_input.to(device).squeeze(1)
            raw_audio_attention_masks = raw_audio_attention_masks.to(device)
            aug_audio_attention_masks = aug_audio_attention_masks.to(device)
            raw_audio_seq_embedding = raw_audio_seq_input_padded.to(device)
            aug_audio_seq_embedding = aug_audio_seq_input_padded.to(device)
            raw_text_seq_embedding, raw_text_cls_embeddings = model1(raw_text)
            aug_text_seq_embedding, aug_text_cls_embeddings = model1(aug_text)

            attention_mask, token_type_ids = create_attention_mask_and_token_type_ids(raw_audio_seq_embedding,
                                                                                      raw_audio_attention_masks,
                                                                                      raw_text_seq_embedding,
                                                                                      max_length=2800)
            logits, _ = model2(raw_audio_seq_embedding, raw_text_seq_embedding, attention_mask, token_type_ids)
            predictions = torch.argmax(logits, dim=-1)
            loss = classification_criterion(logits, labels.to(device))
            total_loss += loss.item()

            correct_predictions += (predictions == labels.to(device)).sum().item()
            total_predictions += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predictions[i] == labels[i]).item()
                class_total[label] += 1

    avg_loss = total_loss / total_predictions
    class_accuracy = class_correct / class_total
    accuracy = (class_accuracy).mean()
    class_weights = class_total / total_predictions
    weighted_accuracy = (class_accuracy * class_weights).sum()
    return accuracy, avg_loss, weighted_accuracy, class_accuracy, class_weights


num_epochs = 30
best_model_path = 'best_emotion_prediction_model.pth'
best_accuracy = 0.0
# 深拷贝初始化模型
base_temperature_model = copy.deepcopy(temperature_model_init).to(device)
base_roberta_model = copy.deepcopy(roberta_model_init).to(device)
base_TextEmbedding_model = copy.deepcopy(TextEmbedding_model_init).to(device)
base_Bert_adapter_multimodel_fusion = copy.deepcopy(Bert_adapter_multimodel_fusion_init).to(device)
base_projection_head = copy.deepcopy(projection_head_init).to(device)


# 循环训练每个 Session
for session_id in range(2, 6):
    print(f"Training on Session {session_id}")
    temperature_model_init.load_state_dict(base_temperature_model.state_dict())
    roberta_model_init.load_state_dict(base_roberta_model.state_dict())
    TextEmbedding_model_init.load_state_dict(base_TextEmbedding_model.state_dict())
    Bert_adapter_multimodel_fusion_init.load_state_dict(base_Bert_adapter_multimodel_fusion.state_dict())
    projection_head_init.load_state_dict(base_projection_head.state_dict())

    optimizer = torch.optim.Adam(
        list(TextEmbedding_model_init.parameters()) + list(Bert_adapter_multimodel_fusion_init.parameters()) + list(
            temperature_model_init.parameters()),
        lr=1e-5
    )

    train_dataset = IEMOCAPDataset(data_list_path=f'../dataset_session{session_id}/train_data.txt',
                                   iemocap_aug_datapath='/home/wangchai/SpeechEmotion/FUXIAN_MMER2023/MMER-main/data/iemocap_aug/out/')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataset = IEMOCAPDataset(data_list_path=f'../dataset_session{session_id}/test_data.txt',
                                 iemocap_aug_datapath='/home/wangchai/SpeechEmotion/FUXIAN_MMER2023/MMER-main/data/iemocap_aug/out/')
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    for epoch in range(num_epochs):
        TextEmbedding_model_init.train()
        Bert_adapter_multimodel_fusion_init.train()
        temperature_model_init.train()

        for batch in tqdm(train_dataloader, desc=f"Session {session_id} Epoch {epoch + 1}/{num_epochs}"):
            raw_audio_seq_input_padded, aug_audio_seq_input_padded, raw_audio_cls_input, aug_audio_cls_input, \
            raw_audio_attention_masks, aug_audio_attention_masks, raw_audio_dimension, aug_audio_dimension, \
            raw_text, aug_text, labels = batch['raw_audio_seq_input_padded'], batch['aug_audio_seq_input_padded'], \
                                         batch['raw_audio_cls_input'], \
                                         batch['aug_audio_cls_input'], batch['raw_audio_attention_masks'], batch[
                                             'aug_audio_attention_masks'], \
                                         batch['raw_audio_dimension'], batch['aug_audio_dimension'], batch['raw_text'], \
                                         batch['aug_text'], batch['label']

            raw_audio_cls_embeddings = raw_audio_cls_input.to(device).squeeze(1)
            aug_audio_cls_embeddings = aug_audio_cls_input.to(device).squeeze(1)
            raw_audio_dimension = raw_audio_dimension.to(device).squeeze(1)
            aug_audio_dimension = aug_audio_dimension.to(device).squeeze(1)
            raw_audio_attention_masks = raw_audio_attention_masks.to(device)
            aug_audio_attention_masks = aug_audio_attention_masks.to(device)
            raw_audio_seq_embedding = raw_audio_seq_input_padded.to(device)
            aug_audio_seq_embedding = aug_audio_seq_input_padded.to(device)
            raw_text_seq_embedding, raw_text_cls_embeddings = TextEmbedding_model_init(raw_text)
            aug_text_seq_embedding, aug_text_cls_embeddings = TextEmbedding_model_init(aug_text)

            attention_mask, token_type_ids = create_attention_mask_and_token_type_ids(raw_audio_seq_embedding,
                                                                                      raw_audio_attention_masks,
                                                                                      raw_text_seq_embedding,
                                                                                      max_length=2800)
            logits1, _ = Bert_adapter_multimodel_fusion_init(raw_audio_seq_embedding, raw_text_seq_embedding, attention_mask,
                                                        token_type_ids)
            classification_loss1 = classification_criterion(logits1, labels.to(device))

            classification_loss = classification_loss1

            # 修改对比损失的计算过程，只对文本头进行映射
            contrastive_loss = 0
            temperature = temperature_model_init()  # 获取当前可学习的温度系数

            for i in range(len(labels)):
                anchor_audio_embedding = raw_audio_cls_embeddings[i]  # 不对语音头进行映射
                anchor_text_embedding =  projection_head_init(raw_text_cls_embeddings[i])  # 只对文本头进行映射
                anchor_label = labels[i]
                anchor_dimension = raw_audio_dimension[i]

                # 找出所有的正样本和负样本
                positive_text_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                negative_text_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
                positive_audio_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                negative_audio_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]

                positive_text_embeddings =  projection_head_init (raw_text_cls_embeddings[positive_text_indices])  # 只对文本头进行映射
                positive_audio_embeddings = raw_audio_cls_embeddings[positive_audio_indices]  # 不对语音头进行映射
                negative_text_embeddings =  projection_head_init (raw_text_cls_embeddings[negative_text_indices])  # 只对文本头进行映射
                negative_audio_embeddings = raw_audio_cls_embeddings[negative_audio_indices]  # 不对语音头进行映射

                # Audio anchor with Text
                if len(positive_text_embeddings) > 1:
                    pos_sim_text = torch.exp(
                        F.cosine_similarity(anchor_audio_embedding.unsqueeze(0),
                                            positive_text_embeddings) / temperature)
                    neg_sim_text = torch.exp(
                        F.cosine_similarity(anchor_audio_embedding.unsqueeze(0),
                                            negative_text_embeddings) / temperature)
                    neg_sim_audio = torch.exp(
                        F.cosine_similarity(anchor_audio_embedding.unsqueeze(0),
                                            negative_audio_embeddings) / temperature)

                    # 计算正样本和负样本的权重
                    pos_weights_text = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), raw_audio_dimension[j].unsqueeze(0), dim=1)
                         for j in positive_text_indices])
                    pos_weights_text = 1.0 / (pos_weights_text + 1e-8)
                    pos_weights_text = pos_weights_text / pos_weights_text.sum()

                    neg_weights_text = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), raw_audio_dimension[j].unsqueeze(0), dim=1)
                         for j in negative_text_indices])
                    neg_weights_text = neg_weights_text / neg_weights_text.sum()

                    pos_sum_text = (pos_sim_text * pos_weights_text.to(device)).sum()
                    neg_sum = (neg_sim_text * neg_weights_text.to(device)).sum() + neg_sim_audio.sum()
                    contrastive_loss += -torch.log(pos_sum_text / (pos_sum_text + neg_sum))

                # Text anchor with Audio
                if len(positive_audio_embeddings) > 1:
                    pos_sim_audio = torch.exp(
                        F.cosine_similarity(anchor_text_embedding.unsqueeze(0),
                                            positive_audio_embeddings) / temperature)
                    neg_sim_audio = torch.exp(
                        F.cosine_similarity(anchor_text_embedding.unsqueeze(0),
                                            negative_audio_embeddings) / temperature)
                    neg_sim_text = torch.exp(
                        F.cosine_similarity(anchor_text_embedding.unsqueeze(0), negative_text_embeddings) / temperature)

                    # 计算正样本和负样本的权重
                    pos_weights_audio = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), raw_audio_dimension[j].unsqueeze(0), dim=1)
                         for j in positive_audio_indices])
                    pos_weights_audio = 1.0 / (pos_weights_audio + 1e-8)
                    pos_weights_audio = pos_weights_audio / pos_weights_audio.sum()

                    neg_weights_audio = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), raw_audio_dimension[j].unsqueeze(0), dim=1)
                         for j in negative_audio_indices])
                    neg_weights_audio = neg_weights_audio / neg_weights_audio.sum()

                    pos_sum_audio = (pos_sim_audio * pos_weights_audio.to(device)).sum()
                    neg_sum = (neg_sim_audio * neg_weights_audio.to(device)).sum() + neg_sim_text.sum()
                    contrastive_loss += -torch.log(pos_sum_audio / (pos_sum_audio + neg_sum))

            contrastive_loss1 = contrastive_loss / (2 * len(labels))

            #######增强数据的对比学习

            contrastive_loss = 0
            temperature = temperature_model_init()  # 获取当前可学习的温度系数

            for i in range(len(labels)):
                anchor_audio_embedding = aug_audio_cls_embeddings[i]  # 不对增强数据的语音头进行映射
                anchor_text_embedding =  projection_head_init (aug_text_cls_embeddings[i])  # 只对增强数据的文本头进行映射
                anchor_label = labels[i]
                anchor_dimension = aug_audio_dimension[i]

                # 找出所有的正样本和负样本
                positive_text_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                negative_text_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
                positive_audio_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                negative_audio_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]

                positive_text_embeddings =  projection_head_init (
                    aug_text_cls_embeddings[positive_text_indices])  # 只对增强数据的文本头进行映射
                positive_audio_embeddings = aug_audio_cls_embeddings[positive_audio_indices]  # 不对增强数据的语音头进行映射
                negative_text_embeddings =  projection_head_init (
                    aug_text_cls_embeddings[negative_text_indices])  # 只对增强数据的文本头进行映射
                negative_audio_embeddings = aug_audio_cls_embeddings[negative_audio_indices]  # 不对增强数据的语音头进行映射

                # Audio anchor with Text
                if len(positive_text_embeddings) > 1:
                    pos_sim_text = torch.exp(
                        F.cosine_similarity(anchor_audio_embedding.unsqueeze(0),
                                            positive_text_embeddings) / temperature)
                    neg_sim_text = torch.exp(
                        F.cosine_similarity(anchor_audio_embedding.unsqueeze(0),
                                            negative_text_embeddings) / temperature)
                    neg_sim_audio = torch.exp(
                        F.cosine_similarity(anchor_audio_embedding.unsqueeze(0),
                                            negative_audio_embeddings) / temperature)

                    # 计算正样本和负样本的权重
                    pos_weights_text = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), aug_audio_dimension[j].unsqueeze(0), dim=1)
                         for j in  positive_text_indices])
                    pos_weights_text = 1.0 / (pos_weights_text + 1e-8)
                    pos_weights_text = pos_weights_text / pos_weights_text.sum()

                    neg_weights_text = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), aug_audio_dimension[j].unsqueeze(0), dim=1)
                         for j in negative_text_indices])
                    neg_weights_text = neg_weights_text / neg_weights_text.sum()

                    pos_sum_text = (pos_sim_text * pos_weights_text.to(device)).sum()
                    neg_sum = (neg_sim_text * neg_weights_text.to(device)).sum() + neg_sim_audio.sum()
                    contrastive_loss += -torch.log(pos_sum_text / (pos_sum_text + neg_sum))

                # Text anchor with Audio
                if len(positive_audio_embeddings) > 1:
                    pos_sim_audio = torch.exp(
                        F.cosine_similarity(anchor_text_embedding.unsqueeze(0),
                                            positive_audio_embeddings) / temperature)
                    neg_sim_audio = torch.exp(
                        F.cosine_similarity(anchor_text_embedding.unsqueeze(0),
                                            negative_audio_embeddings) / temperature)
                    neg_sim_text = torch.exp(
                        F.cosine_similarity(anchor_text_embedding.unsqueeze(0), negative_text_embeddings) / temperature)

                    # 计算正样本和负样本的权重
                    pos_weights_audio = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), aug_audio_dimension[j].unsqueeze(0), dim=1) for j in
                         positive_audio_indices])
                    pos_weights_audio = 1.0 / (pos_weights_audio + 1e-8)
                    pos_weights_audio = pos_weights_audio / pos_weights_audio.sum()

                    neg_weights_audio = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), aug_audio_dimension[j].unsqueeze(0), dim=1) for j in
                         negative_audio_indices])
                    neg_weights_audio = neg_weights_audio / neg_weights_audio.sum()

                    pos_sum_audio = (pos_sim_audio * pos_weights_audio.to(device)).sum()
                    neg_sum = (neg_sim_audio * neg_weights_audio.to(device)).sum() + neg_sim_text.sum()
                    contrastive_loss += -torch.log(pos_sum_audio / (pos_sum_audio + neg_sum))

            contrastive_loss2 = contrastive_loss / (2 * len(labels))

            contrastive_loss = (contrastive_loss1 + contrastive_loss2) / 2

            binary_pairs, binary_pairs_masks, binary_pair_labels = create_binary_classification_pairs(
                raw_audio_seq_embedding, raw_audio_attention_masks, raw_text_seq_embedding, labels)
            binary_pairs_audio = torch.stack([pair[0] for pair in binary_pairs]).to(device)
            binary_pairs_text = torch.stack([pair[1] for pair in binary_pairs]).to(device)
            binary_pair_labels = torch.tensor(binary_pair_labels).to(device)
            binary_pairs_masks = torch.stack(binary_pairs_masks).to(device)

            binary_attention_mask, binary_token_type_ids = create_attention_mask_and_token_type_ids_for_binary(
                binary_pairs_audio, binary_pairs_masks, binary_pairs_text, max_length=2800)
            binary_logits = Bert_adapter_multimodel_fusion_init(binary_pairs_audio, binary_pairs_text, binary_attention_mask,
                                                           binary_token_type_ids)[1]
            binary_classification_loss1 = classification_criterion(binary_logits, binary_pair_labels)

            binary_classification_loss = binary_classification_loss1

            loss = 0.2 * contrastive_loss + 0.5 * classification_loss + 0.3 * binary_classification_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy, avg_loss, weighted_accuracy, class_accuracy, class_weights = evaluate_model(TextEmbedding_model_init,
                                                                                              Bert_adapter_multimodel_fusion_init,
                                                                                              val_dataloader)
        print(
            f"Session {session_id} Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}, Validation Loss: {avg_loss:.4f}, Weighted Accuracy: {weighted_accuracy:.4f}")
        for i in range(num_labels):
            print(f"Class {i} Accuracy: {class_accuracy[i]:.4f}, Weight: {class_weights[i]:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
