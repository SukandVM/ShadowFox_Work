# Sarcasm Detection using BERT <br/>
As part of Task 3 , I have used a pre-trained BERT Model and fine-tuned it using a dataset consisting of an assortment of tweets<br/>
Prediction follows a simple process...<br/>
1. Given statement is pre-processed to remove any special characters or discontinuous sequences.<br/>
2. RandomOverSampling generates additional testcases in the minority class.<br/>
3. For this particular project I have used Bert for sequence classification specifically.<br/>

# Libraries Used: TensorFlow, Transformers, imbLearn, pandas, numpy<br/>
Example of a sarcastic statement: "Yeah, I really love Mondays..."<br/>
Example of non-sarcastic statement: "Steve Harvey has a cute face"<br/>
