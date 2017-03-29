#!/usr/bin/env Rscript
library(ggplot2)
library(dplyr, warn.conflicts=FALSE)
library(readr)
library(ggthemes)

df1 = read_csv("fiji_r9_nano_rocm1.4-ubuntu14.04.5-hip-gpustream.csv"      ,col_names=FALSE)
df2 = read_csv("fiji_r9_nano_rocm1.4-ubuntu14.04.5-ocl-gpustream.csv"      ,col_names=FALSE)
df3 = read_csv("fiji_r9_nano_rocm1.4-ubuntu14.04.5-old-hc-gpustream.csv"   ,col_names=FALSE)

p100 = read_csv("p100_cuda8_ubuntu14.04.5.gpustream.csv", col_names=FALSE)
gtx1080 = read_csv("gtx1080_cuda8_centos7.3.csv", col_names=FALSE)

df = df1 %>% bind_rows(df2) %>% bind_rows(df3)



add_colnames = function(df){
    colnames(df) = c("alg","bw_mb_per_sec","maxv","meanv","minv","sizeof","array_size","total_volume_gb","api")
    df$sizeof = as.factor(df$sizeof)
    df$alg = as.factor(df$alg)
    levels(df$sizeof) = c("float","double")
    return(df)
}

df = add_colnames(df)
df$model = "AMD Fiji R9 Nano"

p100 = add_colnames(p100)
p100$model = "Nvidia Tesla P100"

gtx1080 = add_colnames(gtx1080)
gtx1080$model = "Nvidia GeForce GTX 1080"

gtx1080$api = "CUDA"
p100$api = "CUDA"


df_with_cuda = df %>% bind_rows(p100) %>% bind_rows(gtx1080)


add_with_cuda = df_with_cuda %>% filter(alg == "Add")
adddf = df %>% filter(alg == "Add")

glimpse(adddf)


add_plot = ggplot( adddf ,aes(total_volume_gb ,bw_mb_per_sec/1024.,color=api)) + theme_bw()
add_plot = add_plot + geom_line(size=2) + facet_wrap( ~ sizeof )
add_plot = add_plot + ggtitle("GPU-Stream Add : AMD R9 Fiji Nano (rocm 1.4)")
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


lim_add_wp100_plot = ggplot( add_with_cuda ,aes(total_volume_gb ,bw_mb_per_sec/1024.,color=api,linetype=model)) #+ theme_bw()
lim_add_wp100_plot = lim_add_wp100_plot + geom_line(size=2) + facet_wrap( ~ sizeof )
#lim_add_wp100_plot = lim_add_wp100_plot + ggtitle("GPU-Stream Add : c[:] = b[:] + a[:]")
lim_add_wp100_plot = lim_add_wp100_plot + xlab("used device memory / GB") + ylab(" mean(Bandwidth) / GB/s")
lim_add_wp100_plot = lim_add_wp100_plot + xlim(0,2) + ylim(200,600) 
lim_add_wp100_plot = lim_add_wp100_plot + theme_solarized_2(light = FALSE) +
  scale_colour_solarized("blue")

ggsave("gpu_stream_lim_add_with_nvidia.png",lim_add_wp100_plot,height=3.5)
ggsave("gpu_stream_lim_add_with_nvidia.svg",lim_add_wp100_plot,height=3.5)
