# DRL-Portfolio-Optimization
A portfolio optimization framework leveraging Deep Reinforcement Learning (DRL)

## Motivation
This repo was created during an Independent Study of [Daniel Fudge](https://www.linkedin.com/in/daniel-fudge) with [Professor Yelena Larkin](https://www.linkedin.com/in/yelena-larkin-6b7b361b/) 
as part of a concurrent Master of Business Administration (MBA) and a [Diploma in Financial Engineering](https://schulich.yorku.ca/programs/fnen/)
from the [Schulich School of Business](https://schulich.yorku.ca/).  The study was broken into 3 terms as described 
below.

#### Term 1 - Initial Investigation (Winter 2019)
The first term was a general investigation into the "_Application of Machine Learning to Portfolio Optimization_".  Here 
we reviewed the different aspects of machine learning and their possible applications to Portfolio Optimization.  During 
this investigation we highlighted **Reinforcement Learning (RL)** as an especially promising area to research and 
proposed the development of a Reinforcement Learning framework to better understand its possible applications.  

Please see the 1st term [report](docs/report1.pdf) for the detailed discussion.  

#### Term 2 - Architectural Design (Fall 2019)
##### Udacity Deep Reinforcement Learning Nanodegree
The [1st term report](docs/report1.pdf) identified the Udacity "Deep Reinforcement Learning" 
Nanodegree as a first step toward gaining a better understanding of this topic.  Both the [syllabus](docs/DRL_Nanodegree_Syllabus.pdf)
and the [site](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) provide a detailed 
description of the Nanodegree.  
This Nanodegree involved the developing DRL networks to solve 3 different [Unity ML-Agents environments](https://unity3d.com/machine-learning/).
The solutions to these environments can be found in the following GitHub repositories:
- [Banana Collector](https://github.com/daniel-fudge/banana_hunter)
- [Reacher](https://github.com/daniel-fudge/reinforcement-learning-reacher)
- [Tennis](https://github.com/daniel-fudge/reinforcement-learning-tennis)

##### Architectural Design Report
With the Udacity Nanodegree complete and a greater understanding of DRL obtained, the 2nd term report, found [here](docs/report2.pdf), 
was generated to detail the proposed "_Deep Reinforcement Learning Architecture for Portfolio Optimization_".  This 
report also described the future research required prior to the 3rd term implementation and suggested future work after 
the 3rd term implementation is complete.  

#### Term 3 - Implementation (Winter 2020)
With the problem statement and architecture defined in the [term 2 report](docs/report2.pdf), it will be the task of 
term 3 to fully implement the architecture.  Note that the intent of this implementation is not to advance the 
state-of-the-art in DRL or its application to portfolio optimization.  Instead it is to generate a functioning DRL 
platform for portfolio optimization.  From this conventional implementation, we can experiment with more advanced 
techniques as highlighted in the 2nd term report.   
Term 3 will also deliver a report that summarizes not only the implementation but also the development environment to 
lower the barrier to entry for researchers new to DRL and the deployment of cloud-based applications.   

## Project Settings
This project contains a great deal of user defined settings which are captured in the [settings file](settings.yml).
For a complete description of this file and the included settings, please refer to the associated [wiki page](https://github.com/daniel-fudge/DRL-Portfolio-Optimization/wiki/Settings-File-Format).

## AWS Execution
It is highly recommended that training of the deep learning models be executed on AWS to leverage its GPU instances.
Many other parts may be executed locally to avoid AWS charges but for simplicity we will conduct all stages of the 
project on AWS.  The cost of a simple notebook instance on SageMaker is extremely cheap and cost really only accumulate 
when you train the model (don't forget to shut down when finished!!!).  If you are new to AWS and SageMaker, I recommend 
the AWS SageMarker [tutorial](https://aws.amazon.com/getting-started/tutorials/build-train-deploy-machine-learning-model-sagemaker/).   
Fir instructions on how to setup and run on AWS, please see the associated [Wiki Page]().

## Local Execution
Although it is not recommended to train on a local PC, you may want to run locally to debug.  If so, please see the 
instructions on the associated [Wiki page](https://github.com/daniel-fudge/DRL-Portfolio-Optimization/wiki/Local-PC-Execution).
 
## Data Preparation
The first and one of the most important stage of this project is data preparation.

## License
This code is copyright under the [MIT License](LICENSE).

## Contributions
Please feel free to raise issues against this repo if you have any questions or suggestions for improvement.
