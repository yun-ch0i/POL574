
## Introduction to Text-as-Data in R
## Date: February 12, 2026
## Author: Yun Choi

## Lab adapted from: Christian Baehr, Elisa Wirsching, Lucia Motolinia, 
## Pedro L. Rodriguez, Kevin Munger, Patrick Chester and Leslie Huang.

## Topics:
## - dfm and tf idf
## - examining your corpus
## - regular expressions
## - preText

################################################# Precept 2: Processing Text in R

## 1.1) Working Directory ------------------------------------------------------

## point this to directory containing precept files
setwd("/Users/christianbaehr/Documents/GitHub/POL_574_SP25/")


## 1.2) Package Management -----------------------------------------------------

## only do these once
#install.packages("pacman")
#install.packages("devtools")
#devtools::install_github("matthewjdenny/preText")
# devtools package required to install quanteda from Github 
#remotes::install_github("quanteda/quanteda.textmodels") 

## use pacman instead of loading individual libraries
pacman::p_load(ggplot2, 
               preText,
               quanteda,
               quanteda.textplots,
               quanteda.textstats,
               readtext)



## 1.3) Load Movie Reviews Into a Corpus ---------------------------------------

## load csv with text in "review" column
reviews <- readtext("data/reviews.csv", text_field = "review") |>
  corpus()


## what constitutes a DOCUMENT in this corpus?


## retrieve document level info
reviews.info <- summary(reviews, n=ndoc(reviews)) # document info

sentence_plot <- ggplot(data = reviews.info, aes(x = Sentences)) + 
  geom_histogram() # distribution of the number of sentences

sentence_plot 

## (we could convert to sentence level corpus)
reviews_sent <- corpus_reshape(reviews, to = "sentences")
summary(reviews_sent)

## check if you have the same number of sentences as before
sum(reviews.info$Sentences)

## lets build a document feature matrix
reviews.dfm <- tokens(reviews, 
                      remove_punct = T,
                      remove_symbols = T, 
                      remove_numbers = T,
                      remove_url = T) |> # tokenize, remove punctuation/symbols/numbers/urls
  tokens_remove(stopwords("en")) |> # remove stopwords
  tokens_remove("br") |> # remove "br"
  tokens_wordstem() |> # use the quanteda stemmer
  dfm()

help("stopwords") # quanteda has stopwords for MANY languages

topfeatures(reviews.dfm)

textplot_wordcloud(reviews.dfm, min_count = 5, random_order = F, rotation = 0.25,
                   color = RColorBrewer::brewer.pal(8, "Dark2"))


## weighted dfms
## TF-IDF(t, d) = TF(t, d) × IDF(t)
## scheme_tf = "count" -> TF(t, d) = raw count of term t in document d
## scheme_tf = "prop" -> TF(t, d) = (count of term t in document d) / (total terms in document d)
## IDF(t) = log(N / df_t); N = total number of documents in the corpus, df_t = number of documents containing term t

reviews.dfm_weighted <- dfm_tfidf(reviews.dfm, base = exp(1)) # exp(1)=e^1=e / defaults to "absolute frequency" 
## ALSO DEFAULTS TO LOG_10 !!! 

reviews.dfm_prop <- dfm_tfidf(reviews.dfm,
                              scheme_tf = "prop",
                              base = exp(1)) # relative frequency


topfeatures(reviews.dfm_prop)
topfeatures(reviews.dfm_prop[nrow(reviews.dfm_prop),])

topfeatures(reviews.dfm_weighted)
topfeatures(reviews.dfm_weighted[nrow(reviews.dfm_weighted),]) # why is ordering identical as frequency weighting?


#######################################


## 1.4) IN CLASS ACTIVITY

## Working in pairs, compute the difference-in-means for the RATE AT WHICH
## the term "bad" occurs in positive reviews versus negative reviews

## Hint: retrieve non-zero entries for a single dfm feature column
feature <- dfm_select(reviews.dfm, pattern = "love") # subset dfm to single feature
occurrences <- feature@x # extract the non-zero rows for "love"
r_indices <- feature@i # row indices of non-zero values

pos <- dfm_subset(reviews.dfm, sentiment==1)
neg <- dfm_subset(reviews.dfm, sentiment==0)

bad_pos <- dfm_select(pos, "bad")
bad_neg <- dfm_select(neg, "bad")

bad_pos_rate <- length(bad_pos@x)/nrow(bad_pos)
bad_neg_rate <- length(bad_neg@x)/nrow(bad_neg)

bad_pos_rate - bad_neg_rate

#######################################


## 2.1) Key Word in Context ----------------------------------------------------

kwic_love <- kwic(tokens(reviews), 
                  pattern = "love", # occurrences of "love"
                  valuetype = "fixed", # default 'glob'
                  window = 4) # four words on either side
head(kwic_love)


head(kwic(tokens(reviews), pattern = "hate")) # what is the problem here

# other options
kwic(tokens(reviews), pattern = "hate*", valuetype = "glob") # * matches any number of characters
kwic(tokens(reviews), pattern = "hate?", valuetype = "glob") # ? matches a single characters
kwic(tokens(reviews), pattern = "hat(e|ed|es|ing)", valuetype = "regex")

## 2.2) Collocations -----------------------------------------------------------

reviews.colloc <- textstat_collocations(reviews) # default is bigrams
## how to interpret?
## count_nested: how many times this 3-word collocation appears within longer sequences
## lambda: log-likelihood ratio that measures association strength
## z: statistical significance

## trigrams
reviews.colloc.3 <- textstat_collocations(reviews, size = 3)

## all n-grams between two and four
reviews.colloc.2_4 <- textstat_collocations(reviews, size = 2:4)
reviews.colloc.2_4[1:10 , ]

## 2.3) Regular Expressions ----------------------------------------------------

## it would take an entire course to teach you (or me) all there is know about
## regular expressions
## - RegEx cheatsheet in Github
## - test your regular expression: https://regex101.com/ 

## break this sentence a character vector of words
mysentence <- "We've never lost an American in space, we're sure as hell not gonna lose one on my watch! Failure is not an option."
mysentence <- strsplit(mysentence, split = " ")[[1]]

grep("a", mysentence, value = T) # any element with "a"
grep("^a", mysentence, value = T) # any element STARTING with "a"
grep("[[:punct:]]", mysentence, value = T) # any element with punctuation

grep("^F+[a-z0-9]+ure$", mysentence, value = T)
## starts with F, contains middle characters with a-z and/or 0-9 and ends with "ure"

grep("^F{1}[a-z0-9]{3}ure$", mysentence, value = T) # why does this work?
# F{1} - Exactly 1 uppercase "F"
# [a-z0-9]{3} - Exactly 3 characters that are lowercase letters or digits


## 3.1) Preprocessing Validation with preText ----------------------------------

preprocessed_documents <- factorial_preprocessing(
  reviews,
  use_ngrams = FALSE,
  infrequent_term_threshold = 0.2,
  verbose = TRUE) 

head(preprocessed_documents$choices)
nrow(preprocessed_documents$choices)

preText_results <- preText(preprocessed_documents,
                           distance_method = "cosine",
                           num_comparisons = 20, # Number of random document pairs to compare
                           verbose = TRUE) 

# Cosine_distance = β₀ + β₁(Stemming) + β₂(Stopwords) + β₃(Lowercase) + β₄(Punctuation) + ... + ε
preText_score_plot(preText_results)

## Questions?
