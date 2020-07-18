# SMS_Spam

This aims to distinguish the SMS to be normal or spam message

Looking at the Spam word cloud, some words, such as free, call, pls call and call now, mostly appear in the spam sms

![Image of Spam Words](https://github.com/ccw0530/SMS_Spam/blob/master/spam%20wordcloud.png)

In the model, SKlearn TFID and random forest are used to predict the sms. Also, it has also tested NLTK classifier by 
using Sklearn random forest with the same hyperparameters. 

&nbsp;

It would show that it spends a lot more time to processing.

**Sklearn TFIDvectorizer and Random Forest:** 30.95982789993286 seconds

**NLTK Classifier using Slearn Random Forest:** 1283.372969865799 seconds

Therefore, using SKlearn alone can spped up the processing time

&nbsp;

The accuracy of these two methods are quite similar: 

**Sklearn TFIDvectorizer and Random Forest:** 0.9739292364990689

**NLTK Classifier using Slearn Random Forest:** 0.9590316573556798
