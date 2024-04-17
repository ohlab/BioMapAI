zscore_table = function(table) {
  print("Feature is the rownames")
  table_zscore = apply(table, 1, function(row) scale(row, center = TRUE, scale = TRUE))
  rownames(table_zscore) = colnames(table)
  table_zscore = table_zscore[, colSums(is.na(table_zscore)) != nrow(table_zscore)]
  table_zscore = table_zscore %>% t() %>% as.data.frame()
  return(table_zscore)
  }
                       
calculate_mean <- function(table, group_list) {
  table_zscore <- table %>% t() %>% as.data.frame()  
  table_mean <- data.frame(matrix(0, nrow = length(names(group_list)), 
                                  ncol = length(colnames(table_zscore))))
  rownames(table_mean) <- names(group_list)
  colnames(table_mean) <- colnames(table_zscore)
  
  table_sample <- rownames(table_zscore)
  
  for (name_i in names(group_list)) {
    sample_i <- table_sample[table_sample %in% group_list[[name_i]]]
    table_i <- table_zscore[sample_i, ]
    table_mean[name_i, ] <- colMeans(table_i)
  }
  table_mean =table_mean %>% t() %>% as.data.frame()
  return(table_mean)
}
                       
calculate_p_value <- function(table, group_list) {
  table_zscore = table %>% t() %>% as.data.frame()
  
  table_p <- data.frame(matrix(0, nrow = length(names(group_list)), 
                                  ncol = length(colnames(table_zscore))))
  rownames(table_p) <- names(group_list)
  colnames(table_p) <- colnames(table_zscore)
  
  table_sample <- rownames(table_zscore)
  
  for (name_i in names(group_list)) {
    sample_i <- table_sample[table_sample %in% group_list[[name_i]]]
    sample_rest <- table_sample[!(table_sample %in% group_list[[name_i]])]
    table_i <- table_zscore[sample_i, ]
    table_rest <- table_zscore[sample_rest, ]
    p_value = c()
    for(gene in colnames(table_zscore)){
        p_value_gene = tryCatch({wilcox.test(table_i[,gene], table_rest[,gene])$p.value},error = function(e) {1})
        p_value = c(p_value, p_value_gene)
    }
    p_value = p.adjust(p_value, method = "fdr")
    table_p[name_i, ] = p_value
  }
  table_p = table_p %>% t() %>% as.data.frame()
  return(table_p)
}    
                       
pvalue_table_adjust = function(p_table){
    p_table[p_table < 0.001] = '***'
    p_table[p_table < 0.01 & p_table >= 0.001] = '**'
    p_table[p_table < 0.05 & p_table >= 0.01] = '*'
    p_table[p_table > 0.05] = ''
    p_table[is.na(p_table)] = ''
    return(p_table)
}
group_mean = function(table, annotation){
    table$annotation = annotation
    result = table %>%
        group_by(annotation) %>%
        summarise(across(everything(), ~ mean(., na.rm = TRUE))) %>% 
        as.data.frame()
    rownames(result) = result$annotation
    result$annotation = NULL
    return(result)
}           
calculate_p_vs_Healthy <- function(table, group_list) {
  table_zscore = table %>% t() %>% as.data.frame()
  
  table_p <- data.frame(matrix(0, nrow = length(names(group_list)), 
                                  ncol = length(colnames(table_zscore))))
  rownames(table_p) <- names(group_list)
  colnames(table_p) <- colnames(table_zscore)
  
  table_sample <- rownames(table_zscore)
  
  for (name_i in names(group_list)) {
    sample_i <- table_sample[table_sample %in% group_list[[name_i]]]
    sample_rest <- table_sample[table_sample %in% group_list[['Healthy']]]
    table_i <- table_zscore[sample_i, ]
    table_rest <- table_zscore[sample_rest, ]
    p_value = c()
    for(gene in colnames(table_zscore)){
        p_value_gene = tryCatch({suppressWarnings(wilcox.test(table_i[,gene], table_rest[,gene])$p.value)},
                                error = function(e) {1})
        p_value = c(p_value, p_value_gene)
    }
    p_value = p.adjust(p_value, method = "fdr")
    table_p[name_i, ] = p_value
  }
  table_p = table_p %>% t() %>% as.data.frame()
  return(table_p)
} 