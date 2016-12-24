require(bigSVM)

set.seed(1234)
n <-  10
p <-  10
x <-  matrix(rnorm(100), ncol = p)
y <- rnorm(10)
x.bm <- as.big.matrix(x)
res <- crossprod_bm(x.bm@address, y, 0:9)

crossprod(x[, 1], x)
