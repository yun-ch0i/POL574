# This script will be helpful for HW3, Question 8.2

library(conText)
library(quanteda)

# Load corpus
data("data_corpus_inaugural")
corp <- corpus(data_corpus_inaugural)
# Load pretrained embeddings
glove <- readRDS("data/glove.rds")

# Set document variable
docvars(corp)$party_simple <- ifelse(docvars(corp)$Party == "Democratic", "Dem", "Other")

# Tokenize
toks <- tokens(
  corp,
  remove_punct = TRUE,
  remove_numbers = TRUE
) |>
  tokens_tolower()

# Feature Co-occurrence Matrix (FCM)
fcm_mat <- fcm(
  toks,
  context = "window",
  window  = 5,
  count   = "weighted",
  weights = 1 / (1:5),
  tri     = FALSE
)

# Compute corpus-specific transform
transform_mat <- compute_transform(
  x = fcm_mat,
  pre_trained = glove,
  weighting = "log"
)

# Feature Embedding Matrix (FEM)
fem_mat <- fem(
  x = fcm_mat,
  pre_trained = glove,
  transform = TRUE,
  transform_matrix = transform_mat,
  verbose = TRUE
)
