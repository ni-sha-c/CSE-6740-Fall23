# CSE-6740-Fall23
Course material for CSE 6740 -- Computational Data Analysis (Graduate-level introduction to machine learning)  
East Architecture 123  
TR 12:30 - 1:45  

## 08/22/2023 -- Lecture 1
 Course introduction and logistics  
 HW0 / background review  

## 08/24/2023 -- Lecture 2  
 Empirical Risk Minimization (ERM)  
 PAC Learner Theory  
 Finite hypothesis classes  

## 08/29/2023 -- Lecture 3  
#### Linear Models:  
$\mathscr{H}  = {{h(.,w,b): h(x,w,b) = w\Phi(x) + b, w \in R^d, b \in R}}$  
#### Square Loss:  
$\mathscr{l}(z, h(z,w,b)) = (h(z,w,b) - y)^2$  
#### Minimizing square loss gives **Ordinary Least Squares (OLS)** estimator: $w^{OLS} = (X^TX)^{-1}X^TY$  
#### Proof of Gauss Markov Theorem  
#### $MSE = Variance + Bias^2$  
#### Ridge Regression:  
$minimize {\frac{1}{m} \Vert{XW - Y}\Vert + \lambda\Vert{W}\Vert^2}$



