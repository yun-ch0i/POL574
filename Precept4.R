
## POL 574: Quantitative Analysis IV
## Date: February 26, 2026
## Author: Yun Choi

## Lab adapted from: Christian Baehr, Elisa Wirsching, Lucia Motolinia, 
## Pedro L. Rodriguez, Kevin Munger, Patrick Chester and Leslie Huang.

## Topics:
## - speaker and style distinctiveness
## - regular expressions pt. 2
## - dictionaries

################################################# Precept 4: Supervised Text Analysis I

## load packages
#install.packages("remotes")
#remotes::install_github("kbenoit/quanteda.dictionaries")
pacman::p_load(dplyr, quanteda, quanteda.corpora, 
               quanteda.dictionaries, readtext, stringr, stylest2,
               SentimentAnalysis, sentimentr)


## 1) Predicting authorship using stylest2 -----------------------------------

## we want to build a model to predict who authored a text

## the stylest2 package comes with some example data
data("novels")

## who are the authors?
unique(novels$author)

## step 1 is identify which features are strong out-of-sample predictors of 
## authorship

novels_dfm <- novels |>
  corpus(text_field = "text") |>
  tokens(remove_punct=T,
         remove_symbols=T,
         remove_numbers=T) |>
  dfm()

## we use k-fold cross validation to determine these features and prevent overfitting

## cross validation 
set.seed(123)

s2_cv <- stylest2_select_vocab(dfm = novels_dfm, 
                               smoothing = 0.5,
                               nfold = 5,
                               cutoffs = seq(10, 90, 10))

s2_cv$cutoff_pct_best # cutoff percentile with best performance
s2_cv$cv_missrate_results 
apply(s2_cv$cv_missrate_results, 2, mean) # average miss pct. by percentile

## now pick the subset of features that fall above the optimal cutoff. We will estimate
## the final model over the full data using only these features.

s2_terms <- stylest2_terms(novels_dfm, 
                           cutoff = s2_cv$cutoff_pct_best) # drop words that appear in more than cutoff_pct_best of documents

## estimate the model of speaker identity based on the selected features
s2_fit <- stylest2_fit(novels_dfm, 
                       terms = s2_terms)

## explore output
View(s2_fit)

## most frequently used terms by each author
term_usage <- s2_fit$rate
# rate = (count of term t in author a's texts + smoothing) / (total words by author a + smoothing × vocabulary size)

authors <- rownames(term_usage)

top_terms_by_author <- list()

for (author in authors) {
  # get this author's word usage
  usage_vec <- term_usage[author, ]
  # sort from most frequent to least frequent
  usage_sorted <- sort(usage_vec, decreasing = TRUE)
  # keep top 6 words
  top_terms <- head(usage_sorted, 6)
  # store result
  top_terms_by_author[[author]] <- top_terms
}

top_terms_by_author

## Make authorship predictions
s2_pred <- stylest2_predict(novels_dfm, s2_fit, term_influence = T)

View(s2_pred)

s2_pred$posterior$predicted
s2_pred$posterior$log_probs
s2_pred$posterior$log_prior
s2_pred$term_influence # which words contributed most to assigning the document to a particular author

s2_pred$term_influence %>% arrange(desc(mean_influence))


## process a Pride and Prejudice excerpt to predict author
pride <- gutenbergr::gutenberg_download(gutenberg_id=1342)

set.seed(111)
pred.text <- pride$text[pride$text!=""] %>% # omit empty excepts
  .[sample(1:length(.), 1000)] |> # randomly select 1000
  paste(collapse = " ") |> # collapse to single string
  tokens() |>
  dfm()

docvars(pred.text)["author"] <- "Austen, Jane"

## fit the model using the Pride excerpt
pred <- stylest2_predict(pred.text, s2_fit)
pred$posterior$predicted # predicted author
pred$posterior$log_probs # log probabilities of authorship


## 2) Regular Expressions, cont. ---------------------------------------------

## 2.1) lookaheads and lookbehinds -----------------------------------------
x <- "The United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America. It consists of 50 states, a federal district, five major unincorporated territories, nine Minor Outlying Islands, and 326 Indian reservations."
str_view(tolower(x), "united(?= states)") # positive lookahead
str_view(tolower(x), "(?<!united )states") # negative lookbehind
# (?=...) — positive lookahead (match if followed by pattern)
# (?!...) — negative lookahead (match if NOT followed by pattern)
# (?<=...) — positive lookbehind (match if preceded by pattern)
# (?<!...) — negative lookbehind (match if NOT preceded by pattern)


## 2.2) Selecting features with pattern recognitiion -----------------
data("data_corpus_irishbudgets")

## You can pass a list of words to the "select" parameter in dfm, but using 
## regular expressions can enable you to get all variants of a word
irishbudgets_dfm <- tokens(data_corpus_irishbudgets) %>% 
  dfm() %>% 
  dfm_select(pattern = c("tax|budg|auster"),
             valuetype = "regex")
featnames(irishbudgets_dfm)  


## 3) Dictionaries -----------------------------------------------------------

## load in UK Conservative Manifestos
manifestos <- read.csv("data/conservative_manifestos.csv", stringsAsFactors = F)

## the Laver Garry dictionary was developed to estimate policy positions of UK
## lawmakers. The dictionary has 7 policy levels, and 19 sub-categories
lgdict <- data_dictionary_LaverGarry

## Run the conservative manifestos through this dictionary
manifestos_lg <- manifestos$text |>
  tokens() |> 
  dfm() |>
  dfm_lookup(lgdict) # maps existing features in a DFM to dictionary categories

## what are these policy levels?
View(lgdict)

# how does this look
as.matrix(manifestos_lg)[1:5, 1:5]
featnames(manifestos_lg)

## inspect results graphically 
plot(manifestos$year, 
     manifestos_lg[,"CULTURE.SPORT"],
     xlab="Year", ylab="SPORTS", type="b", pch=19) #b= both points and lines

## plot conservative values trend
plot(manifestos$year, 
     manifestos_lg[,"VALUES.CONSERVATIVE"],
     xlab="Year", ylab="Conservative values", type="b", pch=19)


## 3.2) Harvard IV Dictionary -------------------------------------------------

# install.packages("SentimentAnalysis")
harvard <- DictionaryGI |>
  dictionary()

## run SOTU speeches thru the Harvard dictionary
sotu <- data_corpus_sotu |>
  tokens() |>
  dfm() |>
  dfm_lookup(harvard)

## what are the dimensions?
featnames(sotu)

## net number of positive terms spoken
sotu_positivity <- as.numeric(sotu[,"positive"] - sotu[,"negative"])

year <- docvars(data_corpus_sotu)["Date"][,1] |>
  as.character() |>
  substr(1, 4) |>
  as.numeric()

## plot net positivity of SOTU speeches over time
plot(year, 
     sotu_positivity,
     xlab="Year", ylab="Net Positivity", type="b", pch=19)


## now lets compare to an NLP-based approach for estimating sentiment
sentiments <- sentiment(data_corpus_sotu)

## aggregate by speech
speech_sent <- sentiments |>
  group_by(element_id) |>
  summarize(nlp_positivity = mean(sentiment))

## add the Harvard IV dictionary estimates to compare
speech_sent$harvard_positivity <- sotu_positivity

cor(speech_sent$nlp_positivity, speech_sent$harvard_positivity)
## Pretty bad

## What might be causing this?
sotu_positivity <- as.numeric(sotu[,"positive"] - sotu[,"negative"]) /as.numeric(sotu[,"positive"] + sotu[,"negative"])

cor(speech_sent$nlp_positivity, sotu_positivity)
