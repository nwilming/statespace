# Statespace

State space analysis 
====================

These python modules implement state space analysis a la Mante et al (Nature, 2013). Have a look at the included notebook for an in-depth demonstration.

How does the state space analysis work?
-------------------------------------

The goal of the state space analysis is to find a representation of a high dimensional data set in a few meaningful dimensions. To find these dimensions it uses a regression model that predicts the physiological response we are interested in based on regressors (e.g. experimental conditions). The regression model is performed separately for each unit (here MEG channel, cell, brain area etc.) and timestep. The trick is now to combine the betas of the regression model across units such that we obtain # of time points x # conditions vectors with length # of units. This vector encodes how much each channel contributes to the activity in a specific condition. We can therefore use it to project the high dimensional activity into a 1D space that is related to a particular condition (e.g. meaningful). Do this for 2 conditions and you obtain nice plots. Along the way Mante et al. perform some cleaning of the data based on PCA.
