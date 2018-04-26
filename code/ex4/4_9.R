library(readr)
library(rstan)


data<-read_csv('weather.csv')
gp_data = data.frame(y = data$temperature)
gp_data$x1 <- data$lon
gp_data$x2 <- data$lat
gp_data <- as.list(gp_data)
gp_data$N <- length(data$lat)
grid_predict <- expand.grid(x1 = seq(-131.1, -114.8, 0.4), x2 = seq(40.8, 51.7, 0.4))
gp_data$x1_predict <- grid_predict$x1
gp_data$x2_predict <- grid_predict$x2
gp_data$N_predict  = length(gp_data$x1_predict)
filename = "gp_multiple_regression.stan"

stan_code <-readChar(filename, file.info(filename)$size)

fit <- stan(model_code=stan_code,data=gp_data)

params<-extract(fit)

par(mfrow=c(1, 4))

alpha_breaks=10 * (0:50) / 50 - 5
hist(log(params$alpha), main="", xlab="log(alpha)", col="red", yaxt='n')

beta1_breaks=10 * (0:50) / 50 - 5
hist(log(params$rho1), main="", xlab="log(rho1)", col="red", yaxt='n')

beta2_breaks=10 * (0:50) / 50 - 5
hist(log(params$rho2), main="", xlab="log(rho2)", col="red", yaxt='n')

sigma_breaks=5 * (0:50) / 50
hist(params$sigma, main="", xlab="sigma", col="red", yaxt='n')

readline(prompt="Press [enter] to continue")

probs = c(0.025,0.5,0.975)
cred <- sapply(1:length(gp_data$x_predict), function(n) quantile(params$y_predict[,n], probs=probs))

par(mfrow=c(1, 1))


library(graphics)
image(seq(-131.1, -114.4, 0.4), seq(40.8, 52.1, 0.4), matrix(colMeans(params$f_predict, dims = 1), nrow = 28, byrow = TRUE))

result = apply(matrix(colMeans(params$f_predict, dims = 1), nrow = 41, byrow = TRUE), 2, rev)
x = seq(-131.1, -114.8, 0.4)
y = seq(40.8, 51.7, 0.4)
pointcol <- colorRampPalette(c('red', 'blue'))(20)[as.numeric(cut(data$temperature, breaks = 20))]
attach(data)
filled.contour(x, y, result, color.palette = colorRampPalette(c('darkred', 'darkblue')),
               plot.axes = {points(data$lon, data$lat, pch = 19, col = pointcol)})




matrix(colMeans(params$f_predict, dims = 1), nrow = 28, byrow = TRUE)

heatmap(result, Colv = NA, Rowv = NA)

