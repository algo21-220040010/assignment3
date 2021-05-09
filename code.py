# -*- coding: utf-8 -*-
#prepare the data
import csv
import re
import torch as t
import torch.nn as nn
import numpy as np


flatten = lambda l: [item for sublist in l for item in sublist]

def removePunctuation(text):
    text = re.sub('[^a-zA-Z]',' ',text)
    return text.strip()

train_keyword=[]
train_location=[]
train_text=[]
target=[]

with open('train.csv') as f:
    csvReader=csv.reader(f)
    header=next(csvReader)
    for row in csvReader:
        train_keyword.append(removePunctuation(row[1]))
        train_location.append(removePunctuation(row[2]))
        train_text.append(removePunctuation(row[3]))
        target.append(row[4])

#将target转换为1与-1的浮点数，方便计算loss
for i,x in enumerate(target):
    if x=='0':
        target[i]=float(-1)
    else:
        target[i]=float(1)
target=t.autograd.Variable(t.Tensor(target))   


def data2seq(dataset,word2index):
    dataSplit=[data.split() for data in dataset]
    sentenceLen=max([len(text) for text in dataSplit])
    data=[]
    length=[]
    for text in dataSplit:
        vecText=[0]*sentenceLen
        vecText[:len(text)]=list(
            map(lambda w:word2index[w] if word2index[w] is not None else word2index['unk'], text))
        data.append(vecText)
        length.append(len(text))
    return data,length

def prepareWordDict(train):
    word2index={'unk':0}
    trainSplit=flatten([text.split() for text in train])
    for vo in trainSplit:
        if word2index.get(vo) is None:
            word2index[vo]=len(word2index)                   
    return word2index

def getBatch(dataset,dataL,targetset,batchSize):
    #(batchSize, seqL, embSize)
    for i in range(0,len(dataset),batchSize):
        try:
            inputs=dataset[i:i+batchSize]
            seqL=dataL[i:i+batchSize]
            targets=targetset[i:i+batchSize]
        except:
            inputs=dataset[i:]
            seqL=dataL[i:]
            targets=targetset[i:]
        yield (inputs,seqL,targets)

class RNNclassiy(nn.Module):
    def __init__(self,vocSize,embSize,hiddenSize,nLayers,batchSize):
        super(RNNclassiy,self).__init__()
        self.rnn=nn.RNN(embSize,hiddenSize,nLayers)
        # in linear layer,output feature equal 2 mean two class classify using max
        self.linear=nn.Linear(10,1)
        self.embSize=embSize
        self.nLayers=nLayers
        self.batchSize=batchSize
        self.hiddenSize=hiddenSize
        
    def init_hidden(self):
        hidden = t.autograd.Variable(
            t.zeros(self.nLayers,self.batchSize,self.hiddenSize))
        return hidden
    def init_weight(self):
        pass
    
    def forward(self,inputs,inputsL,hidden):
        inputPack = t.nn.utils.rnn.pack_padded_sequence(inputs, inputsL, batch_first=True,enforce_sorted=False)
        outputPack, hidden = self.rnn(inputPack, hidden)
        unpacked = t.nn.utils.rnn.pad_packed_sequence(outputPack,batch_first=True,total_length=33)
        linearInput=[x[i-1] for x,i in zip(unpacked[0],unpacked[1])]
        linearInput=t.Tensor([x.detach().numpy().tolist() for x in linearInput])
        linearInput=linearInput.contiguous().view(linearInput.size(0),-1)
        return self.linear(linearInput)



trainWordDict=prepareWordDict(train_text)
trainSeq,trainSeqL=data2seq(train_text, trainWordDict)
emb=t.nn.Embedding(len(trainWordDict),128)
trainSeqEmb=t.autograd.Variable(emb(t.LongTensor(trainSeq)))

#training
EMBSIZE=128
HIDDENSIZE=10
NLAYERS=1
BATCHSIZE=5
EPOCH=10
LR=0.01
RESCHEDULED = False
VOCSIZE=len(trainWordDict)

model=RNNclassiy(VOCSIZE, EMBSIZE, HIDDENSIZE, NLAYERS, BATCHSIZE)
criterion=t.nn.MSELoss()
optimizer = t.optim.Adam(model.parameters(), lr=LR)
for epoch in range(EPOCH):
    #totalLoss=0
    losses=[]
    hidden=model.init_hidden()
    for i,batch in enumerate(getBatch(trainSeqEmb, trainSeqL, target, BATCHSIZE)):
        inputs,inputsL,targets=batch[0],batch[1],batch[2]

        if len(inputs) != BATCHSIZE:
            break
        model.zero_grad()
        output=model(inputs,inputsL,hidden)
        
        loss=criterion(output.view(-1), targets)
        losses.append(loss.item())
        loss.backward()
        t.nn.utils.clip_grad_norm(model.parameters(), 0.5) # gradient clipping
        optimizer.step()

        if i > 0 and i % 500 == 0:
            print("[%02d/%d] mean_loss : %0.2f, Perplexity : %0.2f" % (epoch,EPOCH, np.mean(losses), np.exp(np.mean(losses))))
            losses = []
    if RESCHEDULED == False and epoch == EPOCH//2:
        LR *= 0.1
        optimizer = t.optim.Adam(model.parameters(), lr=LR)
        RESCHEDULED = True