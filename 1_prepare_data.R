options(stringsAsFactors = F)
data1 <- read.csv("data/guardian_minimum-wage.csv", encoding = "UTF-8")
data1$topic <- "minimum wage"
data2 <- read.csv("data/guardian_nuclear-energy.csv", encoding = "UTF-8")
data2$topic <- "nuclear energy"

library(quanteda)
library(dplyr)

combined_data <- rbind(data1, data2) %>%
  rename("doc_id" = id, "text" = body_text)

tg_corpus <- corpus(combined_data)
corpus_sentences <- corpus_reshape(tg_corpus, to = "sentences")
View(head(corpus_sentences))
all_sentences <- docvars(corpus_sentences)
all_sentences$text <- as.character(corpus_sentences)
all_sentences$s_id <- docnames(corpus_sentences)
all_sentences$d_id <- docid(corpus_sentences)
View(head(all_sentences))

write.csv(all_sentences, file = "data/tg_sentence_splits.csv", fileEncoding = "UTF-8")

# -----
g_ne <- read.csv("data/guardian_nuclear-energy.csv", encoding = "UTF-8")
View(head(g_ne))
dim(g_ne)
options(stringsAsFactors = F)
tg_sent <- read.csv("data/tg_sentence_splits_pred.csv", encoding = "UTF-8")
dim(tg_sent)
table(tg_sent$topic)
