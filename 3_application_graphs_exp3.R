require(tidyr)
require(dplyr)
require(ggplot2)

# pred_label: 1 con, 2 pro

data <- read.csv("data/tg_sentence_splits_pred_exp3.csv", encoding = "UTF-8")
data$pred_label <- ifelse(data$pred_label == 1, "con", "pro")
# View(data)
cat(colnames(data))

# data %>%
#   mutate("year" = substr(web_publication_date, 0, 4)) %>%
#   select(year, pred_label, full_4_pred_label) %>%
#   group_by(year, pred_label) %>%
#   count(full_4_pred_label)

selected_cols <- c("X1_0_pred_label", "X1_1_pred_label", "X1_2_pred_label", "X1_3_pred_label", "X1_4_pred_label", "X2_0_pred_label", "X2_1_pred_label", "X2_2_pred_label", "X2_3_pred_label", "X2_4_pred_label","X4_0_pred_label", "X4_1_pred_label", "X4_2_pred_label", "X4_3_pred_label", "X4_4_pred_label", "X8_0_pred_label", "X8_1_pred_label", "X8_2_pred_label","X8_3_pred_label", "X8_4_pred_label", "X10_0_pred_label", "X10_1_pred_label", "X10_2_pred_label", "X10_3_pred_label", "X10_4_pred_label", "X50_0_pred_label","X50_1_pred_label", "X50_2_pred_label", "X50_3_pred_label", "X50_4_pred_label", "X100_0_pred_label", "X100_1_pred_label", "X100_2_pred_label", "X100_3_pred_label","X100_4_pred_label", "full_0_pred_label", "full_1_pred_label", "full_2_pred_label", "full_3_pred_label", "full_4_pred_label")

data_selection <- data %>%
  mutate("year" = as.integer(substr(web_publication_date, 0, 4))) %>%
  filter(year >= "2000") %>%
  select(d_id, year, pred_label, all_of(selected_cols))
# View(data_selection)

melted_df <- reshape2::melt(data_selection, id.vars=c("d_id", "year", "pred_label"), measure.vars = selected_cols)
melted_df$k <- unlist(lapply(strsplit(as.character(melted_df$variable), "_"), FUN = function(x) x[1]))
melted_df$i <- unlist(lapply(strsplit(as.character(melted_df$variable), "_"), FUN = function(x) x[2]))
melted_df$value[melted_df$value == "enviroment/health"] <- "environment/health"
melted_df$value[melted_df$value == "reactorsecurity"] <- "reactor safety"
melted_df$value[melted_df$value == "improvement"] <- "innovation"
grouped_df <- melted_df %>%
  group_by(k, i, year, pred_label) %>%
  count(value) 
head(grouped_df, 15)

arg_per_year <- grouped_df %>%
  group_by(k, i, year, pred_label) %>%
  summarise(arg_per_year = sum(n))

shares_df <- grouped_df %>%
  left_join(arg_per_year, by=c("k", "i", "year", "pred_label")) %>%
  mutate(share = n / arg_per_year)

# full, i = 0
plot_df <- shares_df %>%
  filter(k=="full", i==0)

# DEPRECATED area plot: share per year
ggplot(plot_df, aes(x = year, y = share, group=value, fill=value)) +
  geom_area() + facet_wrap(~pred_label) +
  scale_fill_manual(values = paste0(pals::brewer.rdylbu(9), "FF"), name = "year") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# DEPRECATED line plot: stability of predictions
plot2 <- shares_df %>%
  filter(year == "2011") %>%
  group_by(k, pred_label, value) %>%
  summarise(mean_share = mean(share), sd_share = sd(share))

plot2$k <- factor(plot2$k, levels = c("X1", "X2", "X4", "X8", "X10", "X50", "X100", "full"))

ggplot(plot2, aes(k, mean_share, group=pred_label, color=pred_label)) + 
  geom_line() +
  geom_errorbar(aes(ymin=mean_share-sd_share, ymax=mean_share+sd_share), width=.3, position = position_dodge(0.1)) +
  facet_wrap(~value) + theme_bw()


# ----------------------------------
# Paper plots

# 3 years stability over kplot
grouped_df <- melted_df %>%
  group_by(k, i, year, pred) %>%
  count(value) 
head(grouped_df, 15)

arg_per_year <- grouped_df %>%
  group_by(k, i, year) %>%
  summarise(arg_per_year = sum(n))

shares_df <- grouped_df %>%
  left_join(arg_per_year, by=c("k", "i", "year")) %>%
  mutate(share = n / arg_per_year)

plot2 <- shares_df %>%
  mutate(year = as.character(year)) %>%
  filter(year %in% c("2000", "2010", "2020")) %>%
  group_by(k, year, value) %>%
  summarise(mean_share = mean(share), sd_share = sd(share))

plot2$k <- factor(plot2$k, levels = c("X1", "X2", "X4", "X8", "X10", "X50", "X100", "full"))

ggplot(plot2, aes(k, mean_share, group=year, color=year)) + 
  geom_line() +
  scale_color_manual(values=wesanderson::wes_palette("GrandBudapest1", n=3)) +
  geom_errorbar(aes(ymin=mean_share-sd_share, ymax=mean_share+sd_share), width=.3, position = position_dodge(0.1)) +
  facet_wrap(~value) + theme_bw() + ylab("Share") + xlab("Few-shot training set size") + labs(color="Year") +
  theme(legend.position = "bottom") 


# yearly line plots (relative share)
grouped_df <- melted_df %>%
  group_by(k, i, year, pred_label) %>%
  count(value) 

arg_per_year_and_procon <- grouped_df %>%
  group_by(k, i, year, value) %>%
  summarise(arg_per_year = sum(n))

shares_df2 <- grouped_df %>%
  left_join(arg_per_year_and_procon, by=c("k", "i", "year", "value")) %>%
  mutate(share = n / arg_per_year)

plot3 <- shares_df2 %>%
  mutate(year = as.character(year)) %>%
  filter(k == "X100") %>%
  group_by(year, value, pred_label) %>%
  summarise(relative_share = mean(share), sd_share = sd(share))

ggplot(plot3, aes(year, relative_share, group=pred_label, color=pred_label)) + 
  geom_line() +
  # scale_color_grey() +
  scale_color_manual(values=wesanderson::wes_palette("GrandBudapest1", n=2)) +
  geom_errorbar(aes(ymin=relative_share-sd_share, ymax=relative_share+sd_share), width=.3, position = position_dodge(0.1)) +
  facet_wrap(~value) + theme_bw() + ylab("Relative Share") + xlab("Year") + labs(color="Argument") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), legend.position = "bottom") 



# quali checks
data$text[data$pred_label=="pro" & data$full_0_pred_label=="waste" & startsWith(data$web_publication_date, "2020")]
# looks very good to me :)
data$text[data$pred_label=="con" & data$full_0_pred_label=="waste" & startsWith(data$web_publication_date, "2020")]
# this, too!
data$text[data$pred_label=="pro" & data$full_0_pred_label=="weapons" & startsWith(data$web_publication_date, "2020")]
# looks very good to me :)
data$text[data$pred_label=="con" & data$full_0_pred_label=="weapons" & startsWith(data$web_publication_date, "2020")]
