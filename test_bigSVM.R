require(bigSVM)

set.seed(1234)
n <-  20
p <-  10
x <-  matrix(rnorm(n*p), ncol = p)
y <- rnorm(n)
res <- crossprod(x[, 1], x)

x.bm1 <- as.big.matrix(x, type = "double")
x1 <- as.matrix(x.bm1)
object.size(x1)

x.bm2 <- as.big.matrix(x, type = 'float')
x2 <- as.matrix(x.bm2)
object.size(x2)

res1 <- crossprod_bm(x.bm1@address, y, 0:(nrow(x.bm1)-1))
res1

res2 <- crossprod_bm(x.bm2@address, y, 0:(nrow(x.bm2)-1))
res2

