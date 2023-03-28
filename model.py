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