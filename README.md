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

## 08/31/2023 -- Lecture 4
#### Lecture covered the LASSO regression, comparison with Ridge regression, geometrical interpretation and a discussion about best subset selection which is important in fields such as compressed sensing.
#### In lasso regression, shrinkage ($\frac{\sigma_{i}^{2}}{\sigma_{i}^{2} + \lambda}$) is lesser when $\sigma$ (eigenvalue) is higher. As a result, if data varies in one direction, that principal component is given higher importance, rather than uniform weighting of all principal components.
#### Best subset selection: When $m << d$ or when not all features are important, we want to choose a subset of the features. One possibility for feature selection: randomly choose a subset and carry out regression to evaluate the subset. Problem: ${d \choose k}$ combinations for a fixed value of $\ell_0$ norm $k \implies$ Combinatorially too large $\implies$ Infeasible for modest values of $d, k$.
#### We ideally want to solve for $\ell_0$ i.e. best subset selection. However, $\ell_1$ (lasso) is much easier to solve since it is amenable to linear programming methods. Crucial result by [Candes and Tao 2004]: we can get very close to an optimal $\ell_0$ solution by solving the $\ell_1$ problem instead. Result valid given some assumptions on the data.
#### For the ERM upper bounds in case of Ridge and LASSO regression, $r^2$ term in Ridge will grow linearly with $d^2$, which is much faster than $\log(d)$ in LASSO. Hence, upper bound for Ridge is looser compared to LASSO as $d$ grows.



