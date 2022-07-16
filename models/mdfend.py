import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from torch.autograd import Variable
from transformers import BertTokenizer

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, dropout, emb_type ):
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = 9
        self.gamma = 10
        self.num_expert = 5
        self.fea_size =256
        self.emb_type = emb_type
        if(emb_type == 'bert'):
            self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        kernel_size = 4
        num_filter_maps = 384
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.num_expert),
                                      nn.Softmax(dim = 1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings = self.domain_num, embedding_dim = emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num = 1, input_size=emb_dim, output_size=self.fea_size)
        self.classifier = MLP(320, mlp_dims, dropout)
        self.label_conv = nn.Conv1d(768, num_filter_maps, kernel_size=kernel_size, padding=int(math.floor(kernel_size/2)))
        self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
        self.label_fc1.to('cuda')
        self.label_conv.to('cuda')
        self.lmbda = 0.2

    def _compare_label_embeddings(self, target, b_batch):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            if len(bi)>0:
                ti = target[i]
                zi = self.classifier.mlp[4].weight[ti,:]
                diff = (zi - bi).mul(zi - bi).mean()

                #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
                diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs

    def embed_descriptions(self, desc_data, gpu=True):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        tokenizer = BertTokenizer(vocab_file='/root/autodl-nas/MDFEND-Weibo21-MIE/pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
        for inst in desc_data:
            embeded_list = [torch.tensor(tokenizer.encode(inst[i],max_length = 35, 
            padding = 'max_length')).reshape(1,-1).cuda() for i in range(len(inst))]
            if embeded_list == []:
                b_batch.append([])
            else:
                d = self.bert(torch.stack(embeded_list).reshape(-1,35)).last_hidden_state
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
        return b_batch


    def forward(self,batch_size,**kwargs):
        inputs = torch.stack(kwargs['content'])
        masks = torch.stack(kwargs['content_masks'])
        if self.emb_type == "bert":
            init_feature = self.bert(inputs, attention_mask = masks)[0]
        elif self.emb_type == 'w2v':
            init_feature = inputs
        
        feature, _ = self.attention(init_feature, masks)
        domain_embeddings = list()
        for i in range(4):
            domain_embeddings.append(self.domain_embedder(torch.tensor([i]).cuda()).squeeze(1))

        domain_embedding = domain_embeddings[0]
        for i in range(1,4):
            domain_embedding += domain_embeddings[i]
        domain_embedding = domain_embedding.repeat(batch_size,1)
        gate_input_feature = feature
        gate_input = torch.cat([domain_embedding, gate_input_feature], dim = -1)
        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)
        return torch.sigmoid(label_pred)

class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir, 
                 description,
                 mapping,
                 emb_type = 'bert', 
                 loss_weight = [1, 0.006, 0.009, 5e-5],
                 early_stop = 5,
                 epoches = 100,
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        self.emb_type = emb_type
        
        self.description = description
        self.mapping = mapping

        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir

  
    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = MultiDomainFENDModel(self.emb_dim, self.mlp_dims, self.bert, self.dropout, self.emb_type)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.98)
        
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            '''
            0-44症状
            45-60检查
            61-64手术
            65-70一般信息
            '''
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                label_index = [list(set(sample[3])) for sample in batch]
                self.label_index_batch = self.create_label_index_batch(label_index)
                embeded_desc = self.model.embed_descriptions(self.label_index_batch)
                diffs = self.model._compare_label_embeddings(label_index,embeded_desc)
                optimizer.zero_grad()
                label_pred = self.model(len(batch_data['content']),**batch_data)
                loss =  loss_fn(label_pred, torch.stack(label).float()) 
                loss += torch.stack(diffs).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
                
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_mdfend.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')))
        results = self.test(self.test_loader)
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')

    def create_label_index_batch(self, label_index):
        label_index_batch = []
        for instance in label_index:
            instance_list = []
            for i in instance:
                if i >= 0 and i <= 44:
                    instance_list.append(','.join([self.mapping[str(i)]]+self.description['症状'][self.mapping[str(i)]]))
                elif i >= 45 and i <= 60:
                    instance_list.append(','.join([self.mapping[str(i)]]+self.description['检查'][self.mapping[str(i)]]))
                elif i >= 61 and i <= 64:
                    instance_list.append(','.join([self.mapping[str(i)]]+self.description['手术'][self.mapping[str(i)]]))
                elif i >= 65 and i <= 70:
                    instance_list.append(','.join([self.mapping[str(i)]]+self.description['一般信息'][self.mapping[str(i)]]))
            label_index_batch.append(instance_list)
        return label_index_batch

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_label_pred = self.model(len(batch_data['content']),**batch_data)

                label.extend(torch.stack(batch_label).detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, self.category_dict)
