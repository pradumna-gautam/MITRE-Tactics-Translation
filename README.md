# Translating Cybersecurity Descriptions into Interpretable MITRE Tactics using Transfer Learning
## Research Project

## Problem Statement/Introduction

• Intrusion logs and threat intelligence reports have been developed to assist security analysts.

• Description in these logs and reports, however, can be cryptic and not easy to interpret. 

Thus: We ask: Given a description of cyberattack techniques, how to interpret the intended effects (MITRE Tactics)?

• E.g.,1, Initialization scripts can be used to perform administrative functions, which may often execute other programs or send information to an internal logging server.

• E.g.,2, Custom Outlook forms can be created that will execute code when a specifically crafted email is sent.

Is it Privilege Escalation? Persistence? Both?

Solution: 

Developed a Natural Language Processing (NLP) model to translate cybersecurity descriptions into one or more corresponding tactics to assist analysts in diagnosing what adversaries try to accomplish.


## Dataset


• Curated Total of 4500+ Descriptions with their corresponding tactic(s)
(https://attack.mitre.org/)

![alt text](https://user-images.githubusercontent.com/65444978/228394286-485af89f-bc07-4311-940a-a98bc51a2e78.png)

• Pair-wise overlap for MITRE tactic descriptions

• Diagonal values correspond to the single-tactic descriptions

• Some descriptions match to two or more tactics. Hence, the total of 5971 instances are more than the curated descriptions

## Methodology

Used transfer learning on the BERT model since it was pre-trained on a vast amount of text data and has the capacity to learn semantic knowledge from a description bidirectionally.

Build:
• Multi-Label Classification for the total of 14 MITRE Tactics

• Trained model with BERT, SecBERT, and SecureBERT (fine-tuned with various cybersecurity-related text corpuses)

![alt text](https://user-images.githubusercontent.com/65444978/228394269-8105ef8e-3bda-49d6-965f-ed664afda45a.png)


## Results

Results for running the three BERT models with 30 epochs using 5-fold cross-validation.

![alt text](https://user-images.githubusercontent.com/65444978/228394289-e2569718-7d22-4e02-b960-866470b004ec.png)

Results for per-tactic F1 score for the three models to measure the differences in values for single-label and multi-label descriptions.

![alt text](https://user-images.githubusercontent.com/65444978/228394300-7fafbcb7-35d3-44f9-8dcc-1a02be1b8a3a.png)

o The 0.76 Micro F1 score in SecureBERT is promising in capturing semantic features of cybersecurity descriptions and dealing with multi-label data.

o The models could reasonably capture overlapping MITRE tactic descriptions
