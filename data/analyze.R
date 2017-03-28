#!/usr/bin/env Rscript
library(ggplot2)
library(dplyr, warn.conflicts=FALSE)
library(readr)
library(ggthemes)

df1 = read_csv("hip-gpustream.csv"      ,col_names=FALSE)
df2 = read_csv("ocl-gpustream.csv"      ,col_names=FALSE)
df3 = read_csv("old-hc-gpustream.csv"   ,col_names=FALSE)

df = df1 %>% bind_rows(df2) %>% bind_rows(df3)


colnames(df) = c("alg","bw_mb_per_sec","maxv","meanv","minv","sizeof","array_size","total_volume_gb","api")
df$sizeof = as.factor(df$sizeof)
df$alg = as.factor(df$alg)
levels(df$sizeof) = c("float","double")

adddf = df %>% filter(alg == "Add")
glimpse(adddf)
add_plot = ggplot( adddf ,aes(total_volume_gb ,bw_mb_per_sec/1024.,color=api)) + theme_bw()
add_plot = add_plot + geom_line(size=2) + facet_wrap( ~ sizeof )
add_plot = add_plot + ggtitle("GPU-Stream Add on AMD R9 Fiji Nano with rocm 1.4 (Ubuntu 14.04.5)")
add_plot = add_plot + xlab("used device memory / GB") + ylab(" mean(Bandwidth) / GB/s")
## add_plot = add_plot + scale_y_log10() 
## add_plot = add_plot + scale_x_log10() 
ggsave("gpu_stream_add.png",add_plot,height=3.5)
ggsave("gpu_stream_add.svg",add_plot,height=3.5)


lim_add_plot = ggplot( adddf ,aes(total_volume_gb ,bw_mb_per_sec/1024.,color=api)) #+ theme_bw()
lim_add_plot = lim_add_plot + geom_line(size=2) + facet_wrap( ~ sizeof )
lim_add_plot = lim_add_plot + ggtitle("GPU-Stream Add on AMD R9 Fiji Nano (rocm 1.4, Ubuntu 14.04.5)")
lim_add_plot = lim_add_plot + xlab("used device memory / GB") + ylab(" mean(Bandwidth) / GB/s")
lim_add_plot = lim_add_plot + xlim(0,2) + ylim(350,500) 
lim_add_plot = lim_add_plot + theme_solarized_2(light = FALSE) +
  scale_colour_solarized("blue")

ratio.display <- 16/9
ratio.values <- (2*1024)/(max(adddf$bw_mb_per_sec)-min(adddf$bw_mb_per_sec))
#lim_add_plot = lim_add_plot + coord_fixed(3*ratio.values / ratio.display)

## lim_add_plot = lim_add_plot + scale_y_log10() 
## lim_add_plot = lim_add_plot + scale_x_log10() 
ggsave("gpu_stream_lim_add.png",lim_add_plot,height=3.5)
ggsave("gpu_stream_lim_add.svg",lim_add_plot,height=3.5)
