#!/usr/bin/env Rscript
library(ggplot2)
library(dplyr, warn.conflicts=FALSE)
library(readr)
library(ggthemes)

df1 = read_csv("fiji_r9_nano_rocm1.4-ubuntu14.04.5-hip-gpustream.csv"      ,col_names=FALSE)
df2 = read_csv("fiji_r9_nano_rocm1.4-ubuntu14.04.5-ocl-gpustream.csv"      ,col_names=FALSE)
df3 = read_csv("fiji_r9_nano_rocm1.4-ubuntu14.04.5-old-hc-gpustream.csv"   ,col_names=FALSE)

cuda = read_csv("p100_cuda8_ubuntu14.04.5.gpustream.csv", col_names=FALSE)

df = df1 %>% bind_rows(df2) %>% bind_rows(df3)

df_with_cuda = df %>% bind_rows(cuda)


add_colnames = function(df){
    colnames(df) = c("alg","bw_mb_per_sec","maxv","meanv","minv","sizeof","array_size","total_volume_gb","api")
    df$sizeof = as.factor(df$sizeof)
    df$alg = as.factor(df$alg)
    levels(df$sizeof) = c("float","double")
    return(df)
}

df = add_colnames(df)
df_with_cuda = add_colnames(df_with_cuda)

add_with_cuda = df_with_cuda %>% filter(alg == "Add")
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

ggsave("gpu_stream_lim_add.png",lim_add_plot,height=3.5)
ggsave("gpu_stream_lim_add.svg",lim_add_plot,height=3.5)


lim_add_wp100_plot = ggplot( add_with_cuda ,aes(total_volume_gb ,bw_mb_per_sec/1024.,color=api)) #+ theme_bw()
lim_add_wp100_plot = lim_add_wp100_plot + geom_line(size=2) + facet_wrap( ~ sizeof )
lim_add_wp100_plot = lim_add_wp100_plot + ggtitle("GPU-Stream Add\nAMD R9 Fiji Nano (HBM1), Nvidia Tesla P100 (HBM2)")
lim_add_wp100_plot = lim_add_wp100_plot + xlab("used device memory / GB") + ylab(" mean(Bandwidth) / GB/s")
lim_add_wp100_plot = lim_add_wp100_plot + xlim(0,2) + ylim(350,600) 
lim_add_wp100_plot = lim_add_wp100_plot + theme_solarized_2(light = FALSE) +
  scale_colour_solarized("blue")

ggsave("gpu_stream_lim_add_wp100.png",lim_add_wp100_plot,height=3.5)
ggsave("gpu_stream_lim_add_wp100.svg",lim_add_wp100_plot,height=3.5)
