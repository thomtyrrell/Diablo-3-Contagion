dat <- read.table("cont_data.txt",header=TRUE,sep=";")

#################################
# CONTAMINATIONS DISTRIBUTION
#################################
cont <- dat$Contaminations
n <- length(cont)

# Summary Statistics
mu <- mean(cont)
sigma <- var(cont)

# Fitting Gamma distribution
s <- sigma/mu   #scale
a <- mu^2/sigma #shape

# Gamma simulation
gamma_sim <- rgamma(n,a,scale=s)

# Plotting  
qqplot(cont,gamma_sim)
abline(0,1)

cont_hist <- hist(cont, main = "Total Contaminations per Contagion", breaks = seq(min(cont),max(cont),1), ylim = c(0,0.04), plot = TRUE, freq = FALSE)
lines(cont_hist$mids, dgamma(cont_hist$mids,a,scale=s),col="red")

# Goodness-of-fit not so good :(
chisq.test(cont_hist$density,dgamma(cont_hist$mids,a,scale=s))

# This assumes the error between my simulations and the Gamma distribution are normally 
# distributed.  Given the high p-value I would say this assumption is unwarranted.

#################################
# STOP DISTRIBUTION
#################################

stops <- data$Depth
hist(stops,main = "Total Duration per Contagion", breaks = seq(min(stops),max(stops),1), plot = TRUE, freq = FALSE)

