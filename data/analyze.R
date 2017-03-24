#!/usr/bin/env Rscript
library(ggplot2)
library(dplyr, warn.conflicts=FALSE)
library(readr)


df1 = read_csv("hip-gpustream.csv"      ,col_names=FALSE)
df2 = read_csv("ocl-gpustream.csv"      ,col_names=FALSE)
df3 = read_csv("old-hc-gpustream.csv"   ,col_names=FALSE)

df = df1 %>% bind_rows(df2) %>% bind_rows(df3)


colnames(df) = c("alg","bw_mb_per_sec","maxv","meanv","minv","sizeof","array_size","total_volume_gb","api")

head(df)

add_plot = ggplot(df,aes(total_volume_gb,bw_mb_per_sec/1024.,color=api,shape=as.factor(sizeof))) + theme_bw()
add_plot = add_plot + geom_point() 
add_plot = add_plot + ggtitle("GPU-Stream Add on AMD R9 Fiji Nano with rocm 1.4 (Ubuntu 14.04.5)")
add_plot = add_plot + xlab("array size / GB") + ylab(" mean(Bandwidth) / GB/s")
## add_plot = add_plot + scale_y_log10() 
## add_plot = add_plot + scale_x_log10() 
ggsave("gpu_stream_add.png",add_plot)
ggsave("gpu_stream_add.svg",add_plot)
