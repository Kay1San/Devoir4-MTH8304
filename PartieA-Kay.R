library(RColorBrewer) 
library(ggplot2)
theme_set(theme_light(18))
library(cluster)
library(ranger)
library(gridExtra)


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
dir <- "./Fashion-MNIST/"
## lecture des données
mnist.train <- load.image.file(paste(dir, "train-images-idx3-ubyte", sep = ""))
mnist.train.lab <- load.label.file(paste(dir, "train-labels-idx1-ubyte", sep = ""))
## noms des étiquettes de classes
fashion.lab <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                 "Sneaker", "Bag", "Ankle boot")


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
i <- 1
dig.i <- unlist(df.train[i,-785])
dig.xy.i <- data.frame(coord.xy, orig = dig.i)
ggplot(data = dig.xy.i, mapping = aes(x = x, y = y, col = orig)) + geom_point(size=1)+
  scale_colour_gradientn(colors=brewer.pal(n =9, name = "YlGnBu"), name = "") +
  coord_fixed() + xlab("") + ylab("")


lambda <- 10
## descripteurs de l’ensemble d’entraînement à utiliser pour le calcul de la distance
train.loc <- cbind(t(df.train[,-785]), lambda * coord.xy)
## clustering hierarchique - prends quelques minutes à exécuter
hclust.train <- agnes(train.loc)


## extraction d’un clustering d’une taille donnée à partir du clustering hierarchique
k <- 40
clust.k <- cutree(hclust.train, k)
## transformation en variables catégoriques
clust.k.group <- factor(clust.k, levels=1:k)
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


## Création de l’ensemble de validation ayant 5000 exemples
df.valid <- data.frame(x = mnist.train$x[5001:10000,]/255,
                       lab = factor(fashion.lab[mnist.train.lab[5001:10000]+1], levels = fashion.lab))

## Entraînement du classifieur sur les données originales:
## le nombre de clusters = 784
f0 <- ranger(lab~., data = df.train)
pred.f0 <- predict(f0, data = df.valid) ## estimation en validation

## Erreur de classification totale sur l’ensemble de validation
mean(pred.f0$predictions != df.valid$lab)

## Entraînement sur les caractéristiques automatiques
## Le clustering est utilisé pour calculer les intensités moyennes par cluster
## sur l’ensemble d’entraînement:
trainclust.k <- aggregate(t(df.train[,-785]), by = list(clust.k.group), FUN = "mean")
fea.train.k <- data.frame(x = t(trainclust.k[,-1]), lab = df.train$lab)

## Sur l’ensemble de validation:
validclust.k <- aggregate(t(df.valid[,-785]), by = list(clust.k.group), FUN = "mean")
fea.valid.k <- data.frame(x = t(validclust.k[,-1]), lab = df.valid$lab)

## Entraînement du classifieur:
fk <- ranger(lab~., data = fea.train.k)

pred.fk <- predict(fk, data = fea.valid.k) ## estimation en validation

## Erreur de classification totale sur l’ensemble de validation
mean(pred.fk$predictions != fea.valid.k$lab)


############################ EVALUATION 1 ######################################

coord.xy <- (data.frame(x = c(img.y[,28:1]), y = c(img.x[,28:1]))-1)/27
lambdas <- c(2,4,10,20,50)
num_lambdas <- 5

k_lists <- c(20, 40, 60, 80)
num_k <- 4

#Reprensetation de Fashin-MNIST par des images (comme dev 3)
par(mfrow = c(2, 2))
numImages <- 4
for(j in 1:numImages){
  dig <- matrix(mnist.train$x[j,], 28, 28)[, 28:1]
  image(dig, col=brewer.pal(9, "Greys"), main=paste0("Image d'indice ", j))
}

#Representation de Fashion-MNIST en utilisant code du debut
p <- list()
for(i in 1:numImages){
  print(paste("########### image =", i))
  dig.i <- unlist(df.train[i,-785])
  dig.xy.i <- data.frame(coord.xy, orig = dig.i)
  p[[i]] <- ggplot(data = dig.xy.i, mapping = aes(x = x, y = y, col = orig)) + geom_point(size=1)+
    scale_colour_gradientn(colors=brewer.pal(n =9, name = "YlGnBu"), name = "") +
    coord_fixed() + xlab("") + ylab("")
  
  for(j in 1:num_lambdas){
    lambda <- lambdas[j]
    print(paste("##### lambda =", lambda))
    train.loc <- cbind(t(df.train[,-785]), lambda * coord.xy)
    hclust.train <- agnes(train.loc)
    
    for(z in 1:num_k){
      k <- k_lists[z]
      print(paste("## k =", k))
      clust.k <- cutree(hclust.train, k)
      
      ## transformation en variables catégoriques
      clust.k.group <- factor(clust.k, levels=1:k)
      
      ## graphique du clustering
      testP <- ggplot(data = data.frame(coord.xy, class=clust.k.group), mapping = aes(x = x, y = y, col = class)) +
        geom_point(size=5) + 
        scale_colour_manual(values = rep(brewer.pal(n =10, name = "Paired"), ceiling(k/10)), guide = "none") + 
        coord_fixed() + 
        xlab("") + ylab("") + 
        ggtitle(paste("Image", i, "- Lambda", lambda, "- k", k))
      
      file_name <- paste("plot_image", i, "lambda", lambda, "k", k, ".png", sep = "_")
      ggsave(file_name, plot = testP, path = getwd(), width = 6, height = 4)
    }
    
  }
}
do.call(grid.arrange,p)


