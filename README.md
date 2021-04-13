# Building An Intelligent Recommendation System for Personalized Test Scheduling
Code for Building An Intelligent Recommendation System for Personalized Test Scheduling in Computerized Assessments: A Reinforcement Learning Approach (Shin & Bulut, under review) 

## Get Started
### Prerequisites
This code is prepared in python 3. A few python packages are required in order to run the code.
```
numpy == 1.19.5
torch == 1.7.1
torchvision == 0.8.2
pandas == 1.2.2
```
### Components 
You can find the two main folders **main** and **data**
#### Data
This includes *data.xlsx* which provides the example data structure in order to run the model.  
The original data was not shared due to proprietary issues. 
#### Main 
- Utils.py: Includes a several utility function to import and load data etc. 
- Policy+train.py: Includes the main model for training. 

## Acknowledgement 
We would like to thank Dr. Nurakhmetov for his original source code.  
```
@incollection{nurakhmetov2019reinforcement,
  title={Reinforcement Learning Applied to Adaptive Classification Testing},
  author={Nurakhmetov, Darkhan},
  booktitle={Theoretical and Practical Advances in Computer-based Educational Measurement},
  pages={325--336},
  year={2019},
  publisher={Springer, Cham}
}
```
## References 
