require(bigSVM)

set.seed(1234)
n <-  5
p <-  3
x <-  matrix(rnorm(n*p), ncol = p)
y <- rnorm(n)
res <- crossprod(x[, 1], x)
res

x.bm1 <- as.big.matrix(x, type = "double")
x1 <- as.matrix(x.bm1)
object.size(x1)

x.bm2 <- as.big.matrix(x, type = 'float')
x2 <- as.matrix(x.bm2)
object.size(x2)

x.bm3 <- as.big.matrix(x, type = 'integer')

y <- sample(c(1, -1), n, replace = T)
## test bigsvm
bigSVM(x.bm1, y)
# bigSVM(x.bm2, y)
# bigSVM(x.bm3, y)

