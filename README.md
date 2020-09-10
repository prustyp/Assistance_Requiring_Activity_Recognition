# "Recognizing Assistance Requiring Activity via Deep Learning"

## Problem Statement
. There are a large number of people staying in senior living and old age homes. 

. Owing to their age and less mobility they often require assistance by staff which may require constant monitoring. 

. The solution which these senior living provide to the residents is an alarm system which resident could trigger when help is required by the resident. 

. There could be scenarios when an assistance requiring person is incapacitative of raising the alarm for example falling down from wheelchair, 
excessive coughing or sneezing.

. In this project I would like to address this issue by providing a deep learning based solution to recognize an assistance needing activity. 

. I would like to focus mainly on recognizing help needing situations such as falling off a chair, excessive coughing or sneezing. 

. This proposed technique could also be used in hospital so that care-givers could be alerted in case of emergency situations for patients.

## Methodology
In this project, I have employed Recurring Neural Network algorithms to recognize activity occurring in video files. I have primarily focused on 4 different activities: coughing, sneezing, waving hand and falling off chair. The situation of coughing and sneezing might be precursor to some medical emergency. Waving off hand is a simple signal from elderly needing help.

I have collected as much videos under these four categories, so that I could have sufficient amount of training and test videos. There are video repositories available which most activity recognition research community uses to train their model and test such as UCF101 [16], ActivityNet [9], Youtube8M [1] and Kinetics700 [4]. These repositories either have videos or links to videos and they have labeled the actions in the video.

I have prepared the RNN models to train and test each category of video and find out the accuracy of activity recognition. I have also gone over other research work during this time frame to see if there are models/proposed techniques to efficiently identify actions in videos with lesser false positives and true negatives.

## Conclusion
There has been various solutions provided to the elderly to help raise alarm in a senior living home. With this project, I would like to provide a solution to the elderly people to help raise alarm during an emergency situations more efficiently. I have designed an model using pretrained CNN model and RNN to predict four classes of assistance requiring activities with an accuracy of more than 90% for all the models.
