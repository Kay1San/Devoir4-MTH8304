set.seed(1989922) # remplacez par votre matricule
mypath <- "./images_jpeg/train/" # adaptez le chemin à votre ordinateur
img.list <- list.files(path=mypath)
mon.img <- sample(img.list, 1)
img.path <- paste0(mypath, mon.img)

## pour faire le graphique associé à l’image choisie
library(jpeg);library(ggplot2)
imgRGB <- function(img.path){
  img <- readJPEG(img.path)
  imgDm <- dim(img)
  # Assign RGB channels to data frame
  data.frame(
    x = rep(1:imgDm[2], each = imgDm[1]),
    y = rep(imgDm[1]:1, imgDm[2]),
    R = as.vector(img[,,1]),
    G = as.vector(img[,,2]),
    B = as.vector(img[,,3])
  )
}
# ggplot theme to be used
plotTheme <- function() {
  theme(
    panel.background = element_rect(
      linewidth = 3,
      colour = "black",
      fill = "white"),
    axis.ticks = element_line(
      linewidth = 2),
    panel.grid.major = element_line(
      colour = "gray80",
      linetype = "dotted"),
    panel.grid.minor = element_line(
      colour = "gray90",
      linetype = "dashed"),
    axis.title.x = element_text(
      size = rel(1.2),
      face = "bold"),
    axis.title.y = element_text(
      size = rel(1.2),
      face = "bold"),
    plot.title = element_text(
      size = 20,
      face = "bold",
      vjust = 1.5)
  )
}
# Plot the image

img1 <- imgRGB(img.path)

ggplot(data = img1, aes(x = x, y = y)) + geom_point(colour = rgb(img1[c("R", "G", "B")])) +
  labs(title = "Original Image") + xlab("") + ylab("") +
  plotTheme()+ coord_fixed()

library(mclust)
img.mclust <- Mclust(img1) #img.mclust$BIC pour top 3 modeldes

plot(img.mclust, what = "BIC")


i <- 1
col.i <- data.frame(t(apply(img1[c("R", "G", "B")][img.mclust$classification == i,], 2, mean)))
for (i in 2:img.mclust$G){
  col.i <- rbind(col.i,
                 data.frame(t(apply(img1[c("R", "G", "B")][img.mclust$classification == i,], 2, mean))))
}
col.clust <- rgb(col.i)
ggplot(data = img1, aes(x = x, y = y)) +
  geom_point(colour = col.clust[img.mclust$classif]) +
  labs(title = "Segmented Image") +
  xlab("") + ylab("") + plotTheme()+ coord_fixed()



#VVV,4 et VEV,4


img2.mclust <- Mclust(img1, modelName="VVV", G=9)
plot(img2.mclust, what = "BIC")

i <- 1
col2.i <- data.frame(t(apply(img1[c("R", "G", "B")][img2.mclust$classification == i,], 2, mean)))
for (i in 2:img2.mclust$G){
  col2.i <- rbind(col2.i,
                 data.frame(t(apply(img1[c("R", "G", "B")][img2.mclust$classification == i,], 2, mean))))
}
col2.clust <- rgb(col2.i)
ggplot(data = img1, aes(x = x, y = y)) +
  geom_point(colour = col.clust[img2.mclust$classif]) +
  labs(title = "Segmented Image 2") +
  xlab("") + ylab("") + plotTheme()+ coord_fixed()



img3.mclust <- Mclust(img1, modelName="VEV", G=4)
plot(img3.mclust, what = "BIC")

i <- 1
col3.i <- data.frame(t(apply(img1[c("R", "G", "B")][img3.mclust$classification == i,], 2, mean)))
for (i in 2:img3.mclust$G){
  col3.i <- rbind(col3.i,
                  data.frame(t(apply(img1[c("R", "G", "B")][img3.mclust$classification == i,], 2, mean))))
}
col3.clust <- rgb(col3.i)
ggplot(data = img1, aes(x = x, y = y)) +
  geom_point(colour = col.clust[img3.mclust$classif]) +
  labs(title = "Segmented Image 2") +
  xlab("") + ylab("") + plotTheme()+ coord_fixed()


#Si on ne peut pas choisir VVV,4 l'autre modele sera EVV,4

img4.mclust <- Mclust(img1, modelName="EVV", G=4)
plot(img4.mclust, what = "BIC")

i <- 1
col4.i <- data.frame(t(apply(img1[c("R", "G", "B")][img4.mclust$classification == i,], 2, mean)))
for (i in 2:img4.mclust$G){
  col4.i <- rbind(col4.i,
                  data.frame(t(apply(img1[c("R", "G", "B")][img4.mclust$classification == i,], 2, mean))))
}
col4.clust <- rgb(col4.i)
ggplot(data = img1, aes(x = x, y = y)) +
  geom_point(colour = col.clust[img4.mclust$classif]) +
  labs(title = "Segmented Image 4") +
  xlab("") + ylab("") + plotTheme()+ coord_fixed()


##### Element 3 ######

data(faithful)
set.seed(1989922)
N<-5
indices <- sample(nrow(faithful), N)
faithful.data<- faithful[indices, ]

faithful.df <- data.frame(
  x = c(4.117, 3.750, 4.533, 4.033, 4.233),
  y = c(79, 75, 84, 80, 81),
  index = c(226, 98, 262, 34, 141),
  color = c("blue", "black", "black", "blue", "black")
)

hc.single <- hclust(dist(faithful.data), method = "single")
plot(hc.single, main = " Dendrogramme du clustering hiérarchique avec liaison simple ",
      xlab = "", sub = "", cex = .9)

plot(faithful.df$x, faithful.df$y, pch=19, cex=1.5, col=faithful.df$color)
text(faithful.df$x, faithful.df$y, labels=faithful.df$index, pos=3, cex=0.8, col=faithful.df$color)


faithful.df <- data.frame(
  x = c(4.117, 3.750, 4.533, 4.033, 4.233),
  y = c(79, 75, 84, 80, 81),
  index = c(226, 98, 262, 34, 141),
  color = c("blue", "black", "black", "blue", "blue")
)
plot(faithful.df$x, faithful.df$y, pch=19, cex=1.5, col=faithful.df$color)
text(faithful.df$x, faithful.df$y, labels=faithful.df$index, pos=3, cex=0.8, col=faithful.df$color)

faithful.df <- data.frame(
  x = c(4.117, 3.750, 4.533, 4.033, 4.233),
  y = c(79, 75, 84, 80, 81),
  index = c(226, 98, 262, 34, 141),
  color = c("blue", "black", "blue", "blue", "blue")
)
plot(faithful.df$x, faithful.df$y, pch=19, cex=1.5, col=faithful.df$color)
text(faithful.df$x, faithful.df$y, labels=faithful.df$index, pos=3, cex=0.8, col=faithful.df$color)


faithful.df <- data.frame(
  x = c(4.117, 3.750, 4.533, 4.033, 4.233),
  y = c(79, 75, 84, 80, 81),
  index = c(226, 98, 262, 34, 141),
  color = c("blue", "blue", "blue", "blue", "blue")
)
plot(faithful.df$x, faithful.df$y, pch=19, cex=1.5, col=faithful.df$color)
text(faithful.df$x, faithful.df$y, labels=faithful.df$index, pos=3, cex=0.8, col=faithful.df$color)





