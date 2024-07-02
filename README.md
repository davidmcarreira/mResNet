# mResNET (miniResNet)

A Pytorch implementation of a network based on Residual Blocks with computational cost in mind. It outputs the raw logits so it can be used for classification, regression or feature embedding tasks.

# Requirements

The development was made in `Python` `3.10` and `torch` `2.xx`.

# Initialization

The main network is the smaller implementation with 15 layers, but it is prepared to assume other sizes depending on the network's building parametrization.

# TO-DO
- [X] Implement mResNet and test for binary classification.
- [ ] Finish README.md
- [ ] Share training and testing notebooks
- [ ] Add requirements.txt
- [ ] Run tests comparing to MobileNet