library(tidyverse)
library(here)
library(ggplot2)

# read csv with here
df <- read_csv(here("AI_reasoning", "output", "ai_complex_df.csv"))

# mean over last 100 iterations
df_plot <- df %>% 
      group_by(sim_id, judge, reason) %>% 
      slice_tail(n = 10) %>% 
      summarise(across(c1:c10, mean))

df_plot %>%
      ggplot(aes(factor(reason), c10)) +
            geom_boxplot() +
            facet_wrap(~judge)

ggplot(df_plot, aes(reason), c10, color = factor(sim_id))) +
    geom_line(size = 0.1) +
    facet_grid(judge ~ reason) +
    theme_minimal() +
    theme(legend.position = "none") -> p

ggsave(p, here("AI_reasoning", "figs", "sim.png")) 


