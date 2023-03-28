import random

seed_val = 2022
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

total_t0 = time.time()

for i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
      if step % 40 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)

      model.zero_grad()

      result = model(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask,
                      labels=b_labels,
                      return_dict=True)
      
      loss = result.loss
      logits = result.logits

      total_train_loss += loss.item()

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      optimizer.step()
.
      scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        
            result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        loss = result.loss
        logits = result.logits
            
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

#**Summary of the training process**

import pandas as pd
.
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats

## **Performance on the Test Set**
#**Data Preparation**

import pandas as pd

test_df = pd.read_csv('/content/drive/My Drive/Datasets/MITRE_unique_test.csv')

print('Number of test sentences: {:,}\n'.format(test_df.shape[0]))

test_sentences = test_df['MITRE Descriptions'].values
labels_test = test_df.Primary.values
from datasets import ClassLabel

c2lt = ClassLabel(num_classes=14, names=['COLLECTION',
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

test_labels_numerical = [c2lt.str2int(label) for label in labels_test]

input_ids = []
attention_masks = []

for sent in test_sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])


input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(test_labels_numerical)

batch_size = 16  

prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

#**Evaluate on Test set**

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

model.eval()


predictions , true_labels = [], []


for batch in prediction_dataloader:

  batch = tuple(t.to(device) for t in batch)
  

  b_input_ids, b_input_mask, b_labels = batch
  
  with torch.no_grad():

      result = model(b_input_ids, 
                     token_type_ids=None, 
                     attention_mask=b_input_mask,
                     return_dict=True)

  logits = result.logits
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.calibration import calibration_curve
from sklearn.metrics import *

accuracy_set = []
cls_report = []

for i in range(len(true_labels)):
  
  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  
  acc = accuracy_score(true_labels[i], pred_labels_i)
  report = classification_report(true_labels[i], pred_labels_i)                
  accuracy_set.append(acc)
  cls_report.append(report)
  
  flat_predictions = np.concatenate(predictions, axis=0)
print(flat_predictions[:3])
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

flat_true_labels = np.concatenate(true_labels, axis=0)

acc = accuracy_score(flat_true_labels, flat_predictions)
rep = classification_report(flat_true_labels, flat_predictions, target_names =['COLLECTION','COMMAND_AND_CONTROL','CREDENTIAL_ACCESS','DEFENSE_EVASION','DISCOVERY','EXECUTION','EXFILTRATION','IMPACT','INITIAL_ACCESS','LATERAL_MOVEMENT','PERSISTENCE','PRIVILEGE_ESCALATION','RECONNAISSANCE','RESOURCE_DEVELOPMENT'])
#cab = label_ranking_loss(flat_true_labels,flat_predictions)
print('Total Acc: %.3f' % acc)
print(rep)
#print(cab)


## **Saving and Loading Fine-Tuned Model**

import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))