gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

from torch.cuda import is_available
import torch

if torch.cuda.is_available():
    
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available")
    print("GPU: ", torch.cuda.get_device_name(0))
    


import pandas as pd

train_df = pd.read_csv('/content/drive/My Drive/Datasets/MITRE_unique_train.csv')
test_df = pd.read_csv('/content/drive/My Drive/Datasets/MITRE_unique_test.csv')

print(train_df.shape[0])    

train_df.sample(10)

#Extracting Sentences and Lables
sentences = train_df['MITRE Descriptions'].values
labels = train_df.Primary.values

from datasets import ClassLabel

c2l = ClassLabel(num_classes=14, names=['COLLECTION',
                                        'COMMAND_AND_CONTROL',
                                        'CREDENTIAL_ACCESS',
                                        'DEFENSE_EVASION',
                                        'DISCOVERY',
                                        'EXECUTION',
                                        'EXFILTRATION',
                                        'IMPACT',
                                        'INITIAL_ACCESS',
                                        'LATERAL_MOVEMENT',
                                        'PERSISTENCE',
                                        'PRIVILEGE_ESCALATION',
                                        'RECONNAISSANCE',
                                        'RESOURCE_DEVELOPMENT'])

labels_numerical = [c2l.str2int(label) for label in labels]


### **Tokenization**

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Sentences to IDs

input_ids = []

for sentence in sentences:

    encoded_sentence = tokenizer.encode(sentence, add_special_tokens= True)
    input_ids.append(encoded_sentence)

print('Original: ', sentences[118])
print('Token IDs: ', input_ids[118])

#Padding and Truncating

print("Max sentence length: ", max([len(sen) for sen in input_ids]))

from keras_preprocessing.sequence import pad_sequences

MAX_LEN = 128

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', 
                          value=0, truncating='post', padding='post')

#Attention Masks
attention_masks = []

for sent in input_ids:

    attention_mask = [int(token_id>0) for token_id in sent]
    attention_masks.append(attention_mask)

### **Training and Validation Split**

from sklearn.model_selection import train_test_split

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels_numerical, random_state=2022, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels_numerical, random_state=2022, test_size=0.1)

#Converting to Pytorch Data Types

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

### **Train Classification Model**
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels= 14,
                                                      output_attentions= False,
                                                      output_hidden_states= False)
model.cuda()

#### **Optimizer and Learning rate Scheduler**

from transformers import AdamW, BertConfig
optimizer = AdamW(model.parameters(),
                  lr= 2e-5,
                  eps= 1e-8)

from transformers import get_linear_schedule_with_warmup

epochs= 30
total_steps= len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps= 0,
                                            num_training_steps= total_steps)