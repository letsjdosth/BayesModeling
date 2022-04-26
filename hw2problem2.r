sketch <- function(lambda1) {
    lambda2 <- 1 - lambda1
    
    post_lambda_numerator = dnorm(-0.25, 1, sqrt(0.5^2 + 0.1)) * lambda1 + 
        dnorm(-0.25, -1, sqrt(0.5^2 + 0.1)) * lambda2
    post_lambda1 = dnorm(-0.25, 1, sqrt(0.5^2 + 0.1)) * lambda1 / 
        post_lambda_numerator
    post_lambda2 = dnorm(-0.25, -1, sqrt(0.5^2 + 0.1)) * lambda2 / 
        post_lambda_numerator
        
    curve(post_lambda1*dnorm(x, 1.5/14, sqrt(1/14))+ post_lambda2*dnorm(x, -6.5/14, sqrt(1/14))
    ,ylab="density: theta|y", xlim=c(-2, 2), main=paste("lambda1:",lambda1), ylim = c(0, 1.5))
}

dnorm(-0.25, 1, sqrt(0.5^2 + 0.1)) 
dnorm(-0.25, -1, sqrt(0.5^2 + 0.1)) 

par(mfrow = c(2, 3))
sketch(0.999)
sketch(0.99)
sketch(0.95)
sketch(0.9)
sketch(0.75)
sketch(0.5)

