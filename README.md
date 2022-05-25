# Exploring Neuronal Ordinary Differential Equations
## Project in the course DD2424 Deep Learning in Data Science at KTH Royal Insitute of Technology
## By Jannik Wagner, Katrina Liang, Niclas Popp

Neural ordinary differential equations are a class of neural networks with infinitely many layers that generalize residual networks. In this report we present the fundamental theory behind neural ODEs together with two applications. We explain how neural ODEs can be used for descriptive time series modelling and show an example using COVID-19 infections in Sweden. Application two concerns density estimation using a family of generative models called continuous normalizing flows.

## Times Series Analysis

We apply Neural ODEs to epidemic time series data and compare the results to the classical modelling approach using the SEIR equations. The data comprises of the daily new COVID-19 infections in Sweden from 1st November to 20th of April with irregular timestamps. The following figure shows the result of the time series analysis model:

![](read/NeuralODE_realdata.png)

## Continuous Normalizing Flows

First, we applied CNFs to artificial data generated from a two dimensional density based on an image of the flag of Sweden. The GIF shows the contiuous transformation over time of a normal distribution to the learned distribution, using a hypernetwork with depth 3, hpernet width 16, and width 64.

![](CNF/cnf-viz-niter_1000_width64_hidden16cnf-viz-10000.gif)
