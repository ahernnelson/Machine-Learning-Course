library(png)
plotPNG <- function(file){
  img <- readPNG(file)
  grid::grid.raster(img)
}