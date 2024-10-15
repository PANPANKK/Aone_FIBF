import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torchaudio
import numpy as np
import os
import random
class IEMOCAPDataset(Dataset):
    def __init__(self, data_list_path,iemocap_aug_datapath):
        ###raw_data_list_path是我的txt文件；iemocap_aug_datapath存有增强音频的文件夹路径
        self.raw_data_list_path = data_list_path
        self.augmented_audio_path = iemocap_aug_datapath
        ###  列出指定目录下的所有文件和目录名
        self.augmented_audio = os.listdir(self.augmented_audio_path)
        ### 存储原始音频名称与增强音频路径之间的映射关系。
        self.augmented_audio_dictionary = {}
        ### 获取txt文件中的每个原始样本的原始音频路径，文本，情感标签
        with open(data_list_path, 'r', encoding='utf-8') as f: ### 读取txt文件中的原始音频路径，文本，
            self.lines = f.readlines()
        ### 找出每个原始音频对应的所有增强音频的名称，存放到一个字典中。键是原始音频的名称，值是对应的增强音频的路径
        for item in self.augmented_audio:
            gt_audio = "Ses" + item.split("Ses")[-1] ### 获取原始音频的名称
            if gt_audio in self.augmented_audio_dictionary:
                self.augmented_audio_dictionary[gt_audio].append(self.augmented_audio_path + item) ##如果原始音频键已经存在，则将这个
                #对应的增强音频路径，放到对应列表中。
            else:
                self.augmented_audio_dictionary[gt_audio] = [self.augmented_audio_path + item] ##建立原始音频和增强路径的映射
                #{"Ses02F_impro03_F013.wav": ["增强音频路径/Ses04M_script02_2_M_Ses02F_impro03_F013.wav"]}
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        ### 从txt中获取原始音频路径、文本、标签、增强文本
        raw_audio_path, raw_text, label, augmented_text  = self.lines[index].replace('\n', '').split('\t')
        raw_audio_name = os.path.basename(raw_audio_path)
        augmented_wav_index = random.randint(0,len(self.augmented_audio_dictionary[raw_audio_name])-1)
        # 获取增强音频的路径：   iemocap_aug_datapath的路径+/Ses04M_script02_2_M_Ses02F_impro03_F013.wav
        augmented_wav_path = self.augmented_audio_dictionary[raw_audio_name][augmented_wav_index]

        ### 使用torchaudio加载音频，返回音频采样点及采样率
        raw_waveform, raw_sample_rate = torchaudio.load(raw_audio_path)
        aug_waveform, aug_sample_rate = torchaudio.load(augmented_wav_path)
        ### 转化为numpy类型
        raw_waveform = raw_waveform.numpy()
        aug_waveform = aug_waveform.numpy()

        ### audio_input是一个nd.array，维度大小为[batch,1024]，audio_dimension是一个nd.array，维度大小为[batch,3]，使用的是3维情感空间
        raw_audio_seq_input, raw_aduio_cls_input, raw_audio_dimension = process_func(raw_waveform, raw_sample_rate)
        aug_audio_seq_input, aug_aduio_cls_input, aug_audio_dimension = process_func(aug_waveform, aug_sample_rate)

        raw_audio_seq_input= raw_audio_seq_input.squeeze(0)
        aug_audio_seq_input = aug_audio_seq_input.squeeze(0)

        return {
            'raw_audio_seq_input': raw_audio_seq_input,
            'aug_audio_seq_input': aug_audio_seq_input,
            'raw_audio_cls_input': raw_aduio_cls_input,
            'aug_audio_cls_input': aug_aduio_cls_input,
            'raw_sentimental_density': raw_audio_dimension,
            'aug_sentimental_density': aug_audio_dimension,
            'raw_text_input': raw_text,
            'aug_text_input': augmented_text,
            'label': int(label)
        }

def collate_fn(sample_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取一个批次中每个样本的序列长度
    raw_seq_audio_lens = [sample['raw_audio_seq_input'].shape[0] for sample in sample_list]
    aug_seq_audio_lens = [sample['aug_audio_seq_input'].shape[0] for sample in sample_list]

    raw_max_seq_audio_len = max(raw_seq_audio_lens)
    aug_max_seq_audio_len = max(aug_seq_audio_lens)
    max_seq_audio_len = max(raw_max_seq_audio_len, aug_max_seq_audio_len)

    # 用于存放填充后的audio_seq_input
    raw_audio_seq_input_padded = []
    aug_audio_seq_input_padded = []
    # 用于存放填充后的audio_seq_input的mask向量
    raw_audio_attention_masks = []
    aug_audio_attention_masks = []

    for sample in sample_list:
        # 处理原始音频
        raw_seq_len = sample['raw_audio_seq_input'].shape[0]
        raw_pad_len = max_seq_audio_len - raw_seq_len
        # 对音频序列进行填充
        raw_padded_seq = torch.cat([
            sample['raw_audio_seq_input'].to(device),
            torch.zeros((raw_pad_len, sample['raw_audio_seq_input'].shape[1])).to(device)
        ], dim=0)
        raw_audio_seq_input_padded.append(raw_padded_seq)
        # 对掩码进行填充
        raw_attention_mask = torch.cat([
            torch.ones(raw_seq_len).to(device),
            torch.zeros(raw_pad_len).to(device)
        ], dim=0)
        raw_audio_attention_masks.append(raw_attention_mask)

        # 处理增强音频
        aug_seq_len = sample['aug_audio_seq_input'].shape[0]
        aug_pad_len = max_seq_audio_len - aug_seq_len
        # 对音频序列进行填充
        aug_padded_seq = torch.cat([
            sample['aug_audio_seq_input'].to(device),
            torch.zeros((aug_pad_len, sample['aug_audio_seq_input'].shape[1])).to(device)
        ], dim=0)
        aug_audio_seq_input_padded.append(aug_padded_seq)
        # 对掩码进行填充
        aug_attention_mask = torch.cat([
            torch.ones(aug_seq_len).to(device),
            torch.zeros(aug_pad_len).to(device)
        ], dim=0)
        aug_audio_attention_masks.append(aug_attention_mask)

    # 将填充后的序列和掩码堆叠成批次
    raw_audio_seq_input_padded = torch.stack(raw_audio_seq_input_padded)
    aug_audio_seq_input_padded = torch.stack(aug_audio_seq_input_padded)
    raw_audio_attention_masks = torch.stack(raw_audio_attention_masks)
    aug_audio_attention_masks = torch.stack(aug_audio_attention_masks)

    raw_audio_cls_input = torch.stack([sample['raw_audio_cls_input'].to(device) for sample in sample_list])
    aug_audio_cls_input = torch.stack([sample['aug_audio_cls_input'].to(device) for sample in sample_list])
    raw_audio_dimension = torch.stack([sample['raw_sentimental_density'].to(device) for sample in sample_list])
    aug_audio_dimension = torch.stack([sample['aug_sentimental_density'].to(device) for sample in sample_list])
    raw_text = [sample['raw_text_input'] for sample in sample_list]
    aug_text = [sample['aug_text_input'] for sample in sample_list]
    label = torch.tensor([sample['label'] for sample in sample_list], dtype=torch.long).to(device)

    return {
        'raw_audio_seq_input_padded': raw_audio_seq_input_padded,
        'aug_audio_seq_input_padded': aug_audio_seq_input_padded,
        'raw_audio_cls_input': raw_audio_cls_input,
        'aug_audio_cls_input': aug_audio_cls_input,
        'raw_audio_attention_masks': raw_audio_attention_masks,
        'aug_audio_attention_masks': aug_audio_attention_masks,
        'raw_audio_dimension': raw_audio_dimension,
        'aug_audio_dimension': aug_audio_dimension,
        'raw_text': raw_text,
        'aug_text': aug_text,
        'label': label
    }

class RegressionHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
###  加载预训练的Wav2vec2模型，输出的hidden_states1是最后一层隐藏层的沿着时间序列方向平均池化后的结果结果，logits是池化向量映射到三维连续情感空间的结果。
###  如果想要取出，hidden_states的非池化结果，即维度为[batch, seq_len, 1024],只需要取出 hidden_states0 = outputs[0]
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
            attention_mask=None,  # 添加 attention_mask 参数
    ):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)  # 将 attention_mask 传递给 wav2vec2
        hidden_states0 = outputs[0]
        hidden_states1 = torch.mean(hidden_states0, dim=1)
        logits = self.classifier(hidden_states1)
        return hidden_states0, hidden_states1, logits

### 将整个wav2vec计算过程封装，其中processor的作用是对语音进行编码，类似于BERT中的tokenizer.
def process_func(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)
    ### 这里，如果embeddings为true，则返回的y其实是池化后的hidden_states，否则是logits
    with torch.no_grad():
        y = audio_model(y)
    return y[0],y[1],y[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = '../wav2vec2-large-uncased'
processor = Wav2Vec2Processor.from_pretrained(model_name)
audio_model = EmotionModel.from_pretrained(model_name).to(device)
