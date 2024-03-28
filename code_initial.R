load.image.file <- function(filename) {
  ret <- list()
  f <- file(filename, "rb")
  readBin(f, "integer", n = 1, size = 4, endian = "big")
  ret$n <- readBin(f, "integer", n = 1, size = 4, endian = "big")
  nrow <- readBin(f, "integer", n = 1, size = 4, endian = "big")
  ncol <- readBin(f, "integer", n = 1, size = 4, endian = "big")
  x <- readBin(f, "integer", n = ret$n * nrow * ncol, size = 1,
               signed = F)
  ret$x <- matrix(x, ncol = nrow * ncol, byrow = T)
  close(f)
  ret
}
load.label.file <- function(filename) {
  f = file(filename, "rb")
  readBin(f, "integer", n = 1, size = 4, endian = "big")
  n = readBin(f, "integer", n = 1, size = 4, endian = "big")
  y = readBin(f, "integer", n = n, size = 1, signed = F)
  close(f)
  y
}
## remplacer "dir" par le nom du répertoire où se trouvent les données sur votre ordinateur:
dir <- "C:/Users/K-1/Documents/MTH8304/Devoir_4/Fashion-MNIST/"
## lecture des données
mnist.train <- load.image.file(paste(dir, "train-images-idx3-ubyte", sep = ""))
mnist.train.lab <- load.label.file(paste(dir, "train-labels-idx1-ubyte", sep = ""))
## noms des étiquettes de classes
fashion.lab <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                 "Sneaker", "Bag", "Ankle boot")
library(RColorBrewer); library(ggplot2); theme_set(theme_light(18))
## création d’un ensemble d’entraînement ayant 5000 exemples
set.seed(1989922) ## remplacer par son numéro de matricule
shuffle <- sample(mnist.train$n, mnist.train$n, replace = FALSE)
mnist.train$x <- mnist.train$x[shuffle,]
mnist.train.lab <- mnist.train.lab[shuffle]
df.train <- data.frame(x = mnist.train$x[1:5000,]/255,
                       lab = factor(fashion.lab[mnist.train.lab[1:5000]+1], levels = fashion.lab))
## coordonnées des pixels
img.y <- matrix(rep(1:28, 28),28,28, byrow = FALSE)
img.x <- matrix(rep(1:28, 28),28,28, byrow = TRUE)
coord.xy <- (data.frame(x = c(img.y[,28:1]), y = c(img.x[,28:1]))-1)/27
i <- 9
dig.i <- unlist(df.train[i,-785])
dig.xy.i <- data.frame(coord.xy, orig = dig.i)
ggplot(data = dig.xy.i, mapping = aes(x = x, y = y, col = orig)) + geom_point(size=1)+
  scale_colour_gradientn(colors=brewer.pal(n =9, name = "YlGnBu"), name = "") +
  coord_fixed() + xlab("") + ylab("")

library(cluster)
lambda <- 10
## descripteurs de l’ensemble d’entraînement à utiliser pour le calcul de la distance
train.loc <- cbind(t(df.train[,-785]), lambda * coord.xy)
## clustering hierarchique - prends quelques minutes à exécuter
hclust.train <- agnes(train.loc)

## extraction d’un clustering d’une taille donnée à partir du clustering hierarchique
k <- 20
clust.k <- cutree(hclust.train, k)
## transformation en variables catégoriques
clust.k.group <- factor(clust.k, levels=1:k)
test <- data.frame(coord.xy, class=clust.k.group)
## graphique du clustering
ggplot(data = data.frame(coord.xy, class=clust.k.group), mapping = aes(x = x, y = y, col = class)) +
  geom_point(size=5)+ scale_colour_manual(values = rep(brewer.pal(n =10, name = "Paired"),
                                                       ceiling(k/10)), guide = "none") + coord_fixed() + xlab("") + ylab("")

## calcul des intensités moyennes par cluster
dig.i.clust.k <- aggregate(dig.i,by = list(clust.k.group), FUN = "mean")
## attribution de l’intensité moyenne du cluster à chaque pixel du cluster
dig.i.clust.k.merge <- data.frame(cluster = clust.k.group, val = rep(NA, 784))
for (j in 1:k)
  dig.i.clust.k.merge$val[dig.i.clust.k.merge$cluster == j] <- dig.i.clust.k[j,2]
dig.xy.i <- data.frame(coord.xy, orig = dig.i, clust = dig.i.clust.k.merge[,2])
## graphiques des intensités moyennes pour l’image i
ggplot(data = dig.xy.i, mapping = aes(x = x, y = y, col = clust)) + geom_point(size=5)+
  scale_colour_gradientn(colors=brewer.pal(n =9, name = "YlGnBu"), name = "", lim = c(0,1)) +
  coord_fixed()+ xlab("") + ylab("")


library(ranger)
## création de l’ensemble de validation ayant 5000 exemples
df.valid <- data.frame(x = mnist.train$x[5001:10000,]/255,
                       lab = factor(fashion.lab[mnist.train.lab[5001:10000]+1], levels = fashion.lab))
## entraînement du classifieur sur les données originales:
## le nombre de clusters = 784
f0 <- ranger(y~., data = df.train)
pred.f0 <- predict(f0, data = df.valid) ## estimation en validation
## erreur de classification totale sur l’ensemble de validation
mean(pred.f0$pred != df.valid$y)
## entraînement sur les caractéristiques automatiques
## le clustering est utilisé pour calculer les intensités moyennes par cluster
## sur l’ensemble d’entraînement:
trainclust.k <- aggregate(t(df.train[,-785]),by = list(clust.k.group), FUN = "mean")
fea.train.k <- data.frame(x = t(trainclust.k[,-1]), y=df.train$y)
## sur l’ensemble de validation:
validclust.k <- aggregate(t(df.valid[,-785]),by = list(clust.k.group), FUN = "mean")
fea.valid.k <- data.frame(x = t(validclust.k[,-1]), y=df.valid$y)

## entraînement du classifieur:
fk <- ranger(y~., data = fea.train.k)
pred.fk <- predict(fk, data = fea.valid.k) ## estimation en validation
## erreur de classification totale sur l’ensemble de validation
mean(pred.fk$pred != fea.valid.k$y)

