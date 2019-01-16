simp <- function(x,n){
 y <- c()
 for(i in 1:length(x)){ 
   if(x[i] < n){
     y[i] <- (floor(2^n * x[i]) * 2^(-n))
   }
   else {
     y[i] <- (n)
   }
 }
 return(y)
}
x <- seq(0,5,.1)
plot(x,simp(sin(x)+1, 1))

curve(sin(x)+1, add = F, from = 0, to = 6)

for(n in 2:10){ 
  curve(simp(sin(x)+1, 6), add = T, from = 0, to = 6)
}

curve(simp(x-x+1, 3), add = F, from = 0, to = 6)
curve(simp(x, 3), add = T, from = 0, to = 6)
curve(simp(x^2, 3), add = T, from = 0, to = 6)
curve(simp(x^3, 3), add = T, from = 0, to = 6)
curve(simp(1+x^2+x^3, 3), add = F, from = 0, to = 6)
curve(simp(10+x^2, 3), add = F, from = 0, to = 6)
