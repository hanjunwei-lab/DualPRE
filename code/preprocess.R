library(GSVA)

load("../pathway_genes.rdata")
load("../TMEcell.rdata")
TMEcell <- as.data.frame(TMEcell)
pathway_genes[,1] <- gsub(",", "_", pathway_genes[,1])

count2fpkm <- function(count, gene_length_bp) {
  gene_length_kb <- gene_length_bp / 1000
  lib_size_million <- colSums(count) / 1e6
  
  fpkm <- sweep(count, 1, gene_length_kb, "/")
  fpkm <- sweep(fpkm, 2, lib_size_million, "/")
  
  return(log2(fpkm + 1))
}

## gep: Gene expression count matrix (genes × samples)
## gene_length: Gene lengths (bp)
genes <- rownames(gep)
gep <- apply(gep, 2, as.numeric)
rownames(gep) <- genes
na_rows <- apply(gep, 1, function(row) any(is.na(row)))
fpkm_log2 <- count2fpkm(gep, gene_length)

pathway_list<-list()
for (i in 1:length(pathway_genes[,1])) {
  pathway_list[[i]]<-strsplit(pathway_genes[i,2],',')[[1]]
}
names(pathway_list) <- pathway_genes[,1]

cell_list<-list()
for (i in 1:length(TMEcell[,1])) {
  cell_list[[i]]<-strsplit(TMEcell[i,4],',')[[1]]
}
names(cell_list) <- TMEcell[,1]

ssgseaPar <- ssgseaParam(as.matrix(fpkm_log2), pathway_list,minSize=2,normalize = T)
pathway_matrix = gsva(ssgseaPar,verbose=T)

ssgseaPar <- ssgseaParam(as.matrix(fpkm_log2), cell_list,minSize=2,normalize = T)
cell_matrix = gsva(ssgseaPar,verbose=T)

term <- rbind(pathway_matrix,cell_matrix) #term, finally output
