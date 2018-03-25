### STAT MODELING 2, HW 1, EX 1.25
# Load the library
library(car)

# Load the data
prestige = data(Prestige)

# Look at the help file for details
?Prestige

# Get rid of categorical variables
prestige = Prestige[,1:4]

#split data
y = prestige[,4] #using income as dependent variable
x = as.matrix(prestige[,1:3]) #using the three non-categorical predictors

# add an intercept
x = cbind(1,x)

# compute the estimator
betahat = solve(t(x) %*% x) %*% t(x) %*% y

# Fill in the blank
# betacov = ?

# Now compare to lm
# the 'minus 1' notation says not to fit an intercept (we've already hard-coded it as an extra column)
lm1 = lm(y~x-1)
## result 1
summary(lm1)
betacovlm = vcov(lm1)
sqrt(diag(betacovlm))
## result 2
## sqrt(sum square residual /(n-df))
residual <- sqrt(sum((y-x %*% as.matrix(lm1$coefficients))^2)/(dim(x)[1] - dim(x)[2]))
residual*sqrt(diag(solve(t(x) %*% x)))
