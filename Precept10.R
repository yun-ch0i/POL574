
## POL 574: Text as Data
## Date: April 16, 2026
## Author: Yun Choi

## Lab adapted from: Christian Baehr, Elisa Wirsching, Lucia Motolinia, 
## Pedro L. Rodriguez, Kevin Munger, Patrick Chester and Leslie Huang.

## Drawing heavily from:
## https://github.com/prodriguezsosa/conText/blob/master/vignettes/quickstart.md
## (I recommend reading this in full if interested!)

## Topics:
## - Embeddings Regression

### Precept 10: Word Embeddings ------------------------------------------------

## some random processes in the code
set.seed(123)

## Loading packages

pacman::p_load(conText,
               quanteda,
               dplyr,
               text2vec,
               ggplot2)

## 1.1) Prerequisites ----------------------------------------------------------

## What do we need?
## 1) corpus of documents
## 2) PREtrained embeddings
## 3) a transformation matrix associated with the pretrained embeddings

## Embeddings
# glove_link <- "https://drive.google.com/file/d/1ba2sLiTY2w8O2qEHMG6Xe-NEBEluB00A/view?usp=share_link"
glove <- readRDS("data/glove.rds")
dim(glove)
rownames(glove)

## Transformation matrix
khodak <- readRDS("data/khodakA.rds")

## corpus of Congressional Records from conText
corpus <- cr_sample_corpus
docvars <- docvars(corpus) %>% 
  mutate(party_dem = ifelse(party == "D", 1, 0),
         gender_female = ifelse(gender == "F", 1, 0),
         interaction = gender_female * party_dem)
docvars(corpus) <- docvars


## 1.2) Pre-processing ---------------------------------------------------------

## tokenize corpus 

# Step 1: remove stopwords and short tokens
toks_nostop <- corpus %>%
  tokens(remove_punct = T, remove_symbols = T, 
         remove_numbers = T, remove_separators = T) %>%
  tokens_select(pattern = stopwords("en"), selection = "remove", min_nchar = 3)

# Step 2: keep only frequent features (with padding)
toks_nostop_feats <- toks_nostop %>%
  dfm(tolower = T, verbose = F) %>%
  dfm_trim(min_termfreq = 5) %>%
  featnames() %>%
  tokens_select(x = toks_nostop, ., padding = TRUE)

## 1.3) Generating a la carte embeddings ---------------------------------------

## suppose we are interested in differences of the term "immigration" across parties
## build a tokenized corpus of contexts surrounding the target term "immigration"
immig_toks <- tokens_context(x = toks_nostop_feats, 
                             pattern = "immigr*", 
                             window = 6L)

head(immig_toks)
length(immig_toks)
head(docvars(immig_toks), 5)

## build document-feature matrix
immig_dfm <- dfm(immig_toks)
dim(immig_dfm)
immig_dfm[1:3,]

## build a document-embedding-matrix.
## We embed a document by multiplying each of its feature counts with their 
## corresponding pre-trained feature-embeddings, column-averaging the resulting vectors, 
## and multiplying by the transformation matrix.
immig_dem <- dem(x = immig_dfm, 
                 pre_trained = glove, 
                 transform = TRUE, 
                 transform_matrix = khodak, 
                 verbose = TRUE)
dim(immig_dem)
head(immig_dem)

## why cant we just use GloVe embeddings? Want a context-specific embedding for
## EACH occurrence of "immigration"


## 1.4) summarizing local embeddings across groups -----------------------------

## We now have an ALC embedding for each instance of “immigration” in our corpus
## To get a single corpus-wide ALC embedding for “immigration”, 
## we can simply take the column-average of the single-instance ALC embeddings

## to get a single "corpus-wide" embedding, take the column average
immig_wv <- matrix(colMeans(immig_dem), 
                   ncol = ncol(immig_dem)) %>%  `rownames<-`("immigration")
dim(immig_wv)
immig_wv

## to get group-specific embeddings, average within party
immig_wv_party <- dem_group(immig_dem, 
                            groups = immig_dem@docvars$party)
dim(immig_wv_party)
immig_wv_party


## 1.4b ) Substantive differences across groups --------------------------------

immig_nns <- nns(immig_wv_party, 
                 pre_trained = glove, 
                 N = 5, 
                 candidates = immig_wv_party@features, 
                 as_list = TRUE)

## check out results for party
immig_nns[["R"]]
immig_nns[["D"]]

## Cosine similarity
## compute the cosine similarity between each party's embedding and a specific set of features
cos_sim(immig_wv_party, 
        pre_trained = glove, 
        features = c('reform', 'enforcement'), 
        as_list = FALSE)

## Cosine similarity ratio

## we limit candidates to features in our corpus
feats <- featnames(dfm(immig_toks))

## compute ratio
set.seed(2021L)
immig_nns_ratio <- get_nns_ratio(x = immig_toks, 
                                 N = 10,
                                 groups = docvars(immig_toks, 'party'),
                                 numerator = "R",
                                 candidates = feats,
                                 pre_trained = glove,
                                 transform = TRUE,
                                 transform_matrix = khodak,
                                 bootstrap = TRUE,
                                 num_bootstraps = 100,
                                 permute = TRUE,
                                 num_permutations = 10,
                                 verbose = FALSE)

plot_nns_ratio(x = immig_nns_ratio, alpha = 0.01, horizontal = F)



## 1.5) Embedding Regression ---------------------------------------------------
## conText() syntax similar to lm()
## data must be quanteda tokens object with covariates stored as docvars 
## specify a formula consisting of the target word of interest, e.g. “immigration” 
## and the set of covariates. 
## To use all covariates in data, we can specify immigration ~ .. 
## formula can also take vectors of target words:
## e.g. c("immigration", "immigrants") ~ party + gender 
## and phrases:
## e.g. "immigration reform" ~ party + gender 
## (place phrases in quotation marks)

set.seed(123)
model1 <- conText(formula = immigration ~ party + gender,
                  data = toks_nostop_feats,
                  pre_trained = glove,
                  transform = T, transform_matrix = khodak,
                  bootstrap = T, num_bootstraps = 100,
                  permute = T, num_permutations = 100,
                  window = 6, case_insensitive = T,
                  verbose = F)

## why a single coefficient per covariate?
model1@normed_coefficients

## D-dimensional beta coefficients
## the intercept in this case is the ALC embedding for female Democrats
## beta coefficients can be combined to get each group's ALC embedding
DF_wv <- model1['(Intercept)',] # (D)emocrat - (F)emale 
DM_wv <- model1['(Intercept)',] + model1['gender_M',] # (D)emocrat - (M)ale 
RF_wv <- model1['(Intercept)',] + model1['party_R',]  # (R)epublican - (F)emale 
RM_wv <- model1['(Intercept)',] + model1['party_R',] + model1['gender_M',] # (R)epublican - (M)ale 

## nearest neighbors
nns(rbind(DF_wv,DM_wv), N = 10, pre_trained = glove, candidates = model1@features)

nns(rbind(DF_wv,RM_wv), N = 10, pre_trained = glove, candidates = model1@features)

## model with interaction also possible, but need to calculate interaction ourselves
model2 <- conText(formula = immigration ~ party_dem + gender_female + interaction,
                  data = toks_nostop_feats,
                  pre_trained = glove,
                  transform = T, transform_matrix = khodak,
                  bootstrap = T, num_bootstraps = 100,
                  permute = T, num_permutations = 100,
                  window = 6, case_insensitive = T,
                  verbose = F)

## plot the (normed) model coefficients by group 
ggplot(data = model2@normed_coefficients,
       aes(x = factor(coefficient, levels = c("gender_female", "party_dem", "interaction")), y = normed.estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = normed.estimate - 1.96*std.error,
                    ymax = normed.estimate + 1.96*std.error),
                width = 0.5) +
  labs(x = "") +
  theme_bw()

## predicted embedding vector of immigration for female democrat
DF_wv <- model2['(Intercept)',] + model2['gender_female',] + model2['party_dem',] + model2['interaction',]# (D)emocrat - (F)emale 
DM_wv <- model2['(Intercept)',] + model2['party_dem',] # (D)emocrat - (M)ale 
RF_wv <- model2['(Intercept)',] + model2['gender_female',]  # (R)epublican - (F)emale 
RM_wv <- model2['(Intercept)',] # (R)epublican - (M)ale 

## nearest neighbors (in GloVe)
nns(rbind(DF_wv,RM_wv), N = 10, pre_trained = glove, candidates = model2@features)