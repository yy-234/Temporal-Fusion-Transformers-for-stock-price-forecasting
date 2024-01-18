import torch
import torch.nn as nn
import numpy as np
import math

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)
#设置输入数据的长度以及输出数据的长度
input_window = 20
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#Transformer中很重要的一个组件是提出了一种新的位置编码的方式。
# 循环神经网络本身就是一种顺序结构，包含了词在序列中的位置信息。
# 当抛弃循环神经网络结构，完全采用Attention取而代之，这些词序信息就会丢失，模型就没有办法知道每个词在句子中的相对和绝对的位置信息。
# 因此，有必要把词序信号加到词向量上帮助模型学习这些信息，位置编码（PositionalEncoding）就是用来解决这种问题的方法。
# 它的原理是将生成的不同频率的正弦和余弦数据作为位置编码添加到输入序列中，从而使得模型可以捕捉输入变量的相对位置关系。将生成的不同频率的正弦和余弦数据作为位置编码添加到输入序列中，从而使得模型可以捕捉输入变量的相对位置关系
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
       super(PositionalEncoding, self).__init__()
       pe = torch.zeros(max_len, d_model)
       position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)
       pe = pe.unsqueeze(0).transpose(0, 1)
       self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

#model 没有采用原论文中的Encoder-Decoder的架构，
# 将Decoder用了一个全连接层进行代替，用于输出预测值。另外，其中的create_mask将输入进行mask，从而避免引入未来信息。
class TransAm(nn.Module):
    def __init__(self, num_layers=1, feature_size=250, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)#捕捉相对位置关系
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dim_feedforward=49, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()
    def init_weights(self):#初始化权重
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src):

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)# encoder 层
        output = self.decoder(output)
        return output

#窗口函数
#单步预测产生输入和标签

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)
#利用窗口函数分train数据集和test数据集
def get_data():
    series = pd.read_csv('AstraZeneca.csv', usecols=['Close'])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
    train_samples =int(0.7 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]
    return train_sequence, test_data
#数据集处理
def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target
def train(train_data):
    total_loss=0
    model.train()
    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)#mse
        loss.backward()#反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss
def mseloss(loss, data):
    return loss/len(data)

train_data, val_data = get_data()
model = TransAm().to(device)
print(model)
criterion = nn.MSELoss()
lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#优化器
epochs = 200

def evaluate(eval_model, data_source):
    eval_model.eval()#使用测试数据评估
    total_loss = 0
    eval_batch_size = 64
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
        return total_loss / len(data_source)


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    plt.plot(test_result, color="red")
    plt.plot(truth, color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.savefig('./transformer-epoch%d.png' % epoch)
    plt.close()
    return total_loss / i

#迭代
minloss=[]
for epoch in range(1, epochs + 1):
    train(train_data)
    #if (epoch % 10 is 0):
        #plot_and_loss(model, val_data, epoch)
    val_loss = evaluate(model, val_data)
    minloss.append(val_loss)

    #print('-' * 89)
    print('| end of epoch {:3d} |  valid loss {:5.5f} '.format(epoch, val_loss))
    #print('-' * 89)
minloss.sort()
print('min_loos={:5.5f}'.format(minloss[0]))
















