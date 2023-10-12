# EIT_trajectory_prediction_VAE-LSTM
Predicting object motion by using a VAE and LSTM.

## Architecture of the VAE-LSTM Model

If the training is successful, the model should be able to reconstruct voltage data more precisely because a memory is included in the VAE.

![LSTM_VAE](images/VAE_LSTM.png)

- $N$ is the number of training samples
- The memory of the LSTM is $4$ in this case.
- $192$ is the dimension of the voltage vector