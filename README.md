# Translating Cybersecurity Descriptions into Interpretable MITRE Tactics using Transfer Learning

## Introduction

• Intrusion logs and threat intelligence reports have been developed to assist security analysts 
• Description in these logs and reports, however, can be cryptic and not easy to interpret. 

Thus: We ask: Given a description of cyberattack techniques, how to interpret the intended effects (MITRE Tactics)?

• E.g.,1, Initialization scripts can be used to perform administrative functions, which may often execute other programs or send information to an internal logging server.

• E.g.,2, Custom Outlook forms can be created that will execute code when a specifically crafted email is sent.

Is it Privilege Escalation? Persistence? Both?

Developed a Natural Language Processing (NLP) model to translate cybersecurity descriptions into one or more corresponding tactics to assist analysts in diagnosing what adversaries try to accomplish.


## Dataset


• Curated Total of 4500+ Descriptions with their corresponding tactic(s)
(https://attack.mitre.org/)

![alt text](https://drive.google.com/file/d/1BtOq0Tdz1sRkZc3Gu_3i4ErpIOwtFoq1/view?usp=sharing)

• Pair-wise overlap for MITRE tactic descriptions
• Diagonal values correspond to the single-tactic descriptions
• Some descriptions match to two or more tactics. Hence, the total of 5971 instances are more than the curated descriptions

## Methodology

Used transfer learning on the BERT model since it was pre-trained on a vast amount of text data and has the capacity to learn semantic knowledge from a description bidirectionally.

Build:
• Multi-Label Classification for the total of 14 MITRE Tactics
• Trained model with BERT, SecBERT, and SecureBERT (fine-tuned with various cybersecurity-related text corpuses)

![alt text](https://drive.google.com/file/d/1AB0u1osvpMLSloTQRnKqtpWdpjHcKoYf/view?usp=sharing)


## Results

Results for running the three BERT models with 30 epochs using 5-fold cross-validation.

![alt text](https://drive.google.com/file/d/1JYeS5p1A7Pnb3Z5mMeKpEfor69Pr5X0B/view?usp=sharing)

Results for per-tactic F1 score for the three models to measure the differences in values for single-label and multi-label descriptions.

![alt text](https://drive.google.com/file/d/1bQ8ey_tMyOApaZ6WbHsOlH74-bca3OL5/view?usp=sharing)

o The 0.76 Micro F1 score in SecureBERT is promising in capturing semantic features of cybersecurity descriptions and dealing with multi-label data.
o The models could reasonably capture overlapping MITRE tactic descriptions