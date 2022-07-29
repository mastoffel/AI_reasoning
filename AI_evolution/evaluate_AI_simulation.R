library(tidyverse)
library(here)
library(ggplot2)
source("theme_simple.R")

# read csv with here
df <- read_csv(here("AI_evolution", "output", "ai_complex_df_rec.csv"))

# mean over last 100 iterations
df_plot <- df %>% 
      mutate(sim_num = rep(rep(1:10, each = 500), nrow(.) / 5000)) %>%
      group_by(sim_num, iter, judge, reason) %>%
      #slice_tail(n = 100) %>% 
      summarise(across(c1:c10, mean))

df_plot %>%
      ggplot(aes(factor(reason), c10)) +
      geom_boxplot() +
      facet_wrap(~judge, nrow = 1) #+
      #scale_y_log10()

p1 <- df %>%
  ggplot(aes(iter, c10)) +
  geom_line(size = 0.01, alpha = 0.3) + # 
  theme_simple(axis_lines = TRUE, grid_lines = FALSE) +
  facet_grid(judge ~ reason,
             labeller = labeller(.rows = label_both, .cols = label_both)) +
  scale_x_continuous(breaks = c(100, 300, 500)) +
  #scale_y_continuous(breaks = c(1, 5, 10)) +
  scale_y_log10(breaks = c(0.01, 1, 100, 10000),
                labels = c("0.01", "1", "100", "10000")) +
  ylab("Mean utility over traits") +
  xlab("Iteration")
#p1

ggsave("AI_evolution/figs/AI_evo_utility_reinv.jpg", width = 6, height = 5)

