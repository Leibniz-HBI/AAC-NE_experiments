data <- read.csv("fs_results.csv")
# View(data)

library("ggplot2")
ggplot(data, aes(x = n_shot, y = precision, group = label)) +
  stat_summary(geom = "line", fun = mean) +
  facet_wrap(~label)
# fällt ab

ggplot(data, aes(x = n_shot, y = recall, group = label)) +
  stat_summary(geom = "line", fun = mean) +
  facet_wrap(~label)
# steigt an

ggplot(data, aes(x = n_shot, y = f1.score, group = label)) +
  stat_summary(geom = "line", fun = mean) +
  facet_wrap(~label)
# bleibt eher unverändert
