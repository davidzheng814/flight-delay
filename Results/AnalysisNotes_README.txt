

ROC Curves Were Generated As Follows:
- for predicting delays within k hours, I took the vector of times T and turned it into a binary vector where T[i] = 1 if the time is greater than k and T[i] = 0 if the time is less than k
- I then computed the exponential CDF at value k to get the vector of probabilities
- This is basically saying that for k hours, we want to find the probability as stated by our model that there won’t be any delays within the next k hours (no delay is “positive” from a binary classification standpoint)


FinalBetas.json
- included within each folder, contains the beta values
- I standardized the data before training (made each column have zero mean and unit variance), so the betas can directly be compared with each other
- In the writeup its worth noting which features have very positive betas, which features have very negative betas, and which features have beta = 0

Regularization:
- No Regularization and Regularization = 1 did very well
- Regularization = 10 performed terribly but thats expected
- Note this somewhere in the writeup?