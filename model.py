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
