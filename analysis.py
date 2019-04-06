import pandas as pd

# scipy statsmodels workaround
# from https://github.com/statsmodels/statsmodels/issues/3931
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

import statsmodels.api as sm
import statsmodels.formula.api as smf

import pylab as pl
import numpy as np

# use cov_type "HC3" when fitting models

# read the data in
df = pd.read_csv("./merged_data_misleading_science.csv")


print " "
print "Misleading Science Journalism v1 data analysis"


#total number of respondents across conditions:
print "Total number of respondents across conditions: " + str(df.shape[0])


#number pf other gender datapoints
print "Number of 'Other' gender respondents: " + str(df[df.gender == "Other"].shape[0])

#remove other gender respondents
print "Removing other gender respondents"
df = df[df.gender != "Other"]

#new total number of respondents across conditions:
print "New total number of respondents across conditions (after removal): " + str(df.shape[0])

print "Coding likerts and categoricals...."

#coding for likerts

sure_likert = {'Extremely sure': 5, 'Mostly sure': 4, 'Somewhat sure': 3, 'A little sure': 2, 'Not very sure': 1, "Not sure at all (No confidence in the answer)": 0}
accuracy_likert = {'Extremely sure': 5, 'Mostly sure': 4, 'Somewhat sure': 3, 'A little sure': 2, 'Not very sure': 1, "Not sure at all (No confidence in the answer)": 0}
media_accuracy_likert = {'Extremely accurate': 5, 'Mostly accurate': 4, 'Somewhat accurate': 3, 'A little accurate': 2, 'Mostly inaccurate': 1, "Extremely inaccurate": 0}

#coding for some categoricals: 

yes_no_coding = {'Yes': 1, 'No': 0}
gender_coding = {'Female': 1, 'Male': 0}
funding_coding = {'Preventing suicide among teenage girls': 1, 'Preventing suicide among teenage boys': 0}
pp_coding = {'Democratic Party': 'dem', 'Green Party': 'green', 'Republican Party': 'gop', 'Libertarian Party': 'libertarian', 'Other': 'other', 'None, Independent': 'indep'}

# mostly_net_calories -> 0, mostly_quality -> 1
weight_q_coding = {'Losing weight depends MOSTLY on the difference between the number of calories you consume (take in) and the number you burn (use), which is the overall net quantity of calories you consume, REGARDLESS of the quality of food you eat.': 0, 'Losing weight depends MOSTLY on the quality of the food you eat, REGARDLESS of the difference between the number of calories you consume (take in) and the number you burn (use), which is the overall net quantity of calories you consume.': 1}

# male_rates_higher -> 0, female_rates_higher -> 1
suicide_q_coding = {'The rate of suicide among teenage boys': 0, 'The rate of suicide among teenage girls': 1}

# same as above + no_prior_belief -> 2
weight_prior_coding = {'Losing weight depends MOSTLY on the difference between the number of calories you consume (take in) and the number you burn (use), which is the overall net quantity of calories you consume, REGARDLESS of the quality of food you eat.': 0, 'Losing weight depends MOSTLY on the quality of the food you eat, REGARDLESS of the difference between the number of calories you consume (take in) and the number you burn (use), which is the overall net quantity of calories you consume.': 1, 'No prior knowledge.': 2}

# same as above + no prior belief -> 2, rates same -> 3
suicide_prior_coding = {'The rate of suicide among teenage boys': 0, 'The rate of suicide among teenage girls': 1, 'No prior knowledge': 2, 'Neither (rates same)': 3}

df.replace({'confidence_weight': sure_likert, 'confidence_suicide': sure_likert,
  			'accuracy_journalism': media_accuracy_likert, 'gender': gender_coding,
  			'fda_reg': yes_no_coding, 'change_mind': yes_no_coding,
  			'funding': funding_coding, 'weight': weight_q_coding, 
  			'rate_of_suicide': suicide_q_coding, 'prev_rate_belief': suicide_prior_coding,
  			'prev_weight_belief': weight_prior_coding, 'political_party': pp_coding}, inplace=True)

dummy_pp = pd.get_dummies(df['political_party'], prefix='pp')

dummy_pp = dummy_pp.drop(columns=['pp_indep', 'pp_other'])

pp_cols = ["pp_gop", "pp_dem"]

#for pp_val in pp_coding.values():
#	if pp_val != "indep" and pp_val != "other":
#		pp_cols.append("pp_" + pp_val)

print dummy_pp.head()

df = df.join(dummy_pp)

df = df.drop(columns=['Unnamed: 0', 'political_party', 'ResponseId', 'prev_rate_belief', 'prev_weight_belief'])

#df['intercept'] = 1.0

print "Done coding likerts and categoricals."

print " "

print "Setting up new df for dailybeast"

dailybeast_df = df[df.is_nyt != 1]

dailybeast_cols = ['confidence_suicide', 'accuracy_journalism', 'gender', 'funding', 'rate_of_suicide', 'is_dailybeast'] + pp_cols

dailybeast_df = dailybeast_df[dailybeast_cols]

dailybeast_df = dailybeast_df.apply(pd.to_numeric, errors="coerce")

print "dailybeast describe control group"
print dailybeast_df[dailybeast_df.is_dailybeast != 1].describe()

print "dailybeast describe experiment group"
print dailybeast_df[dailybeast_df.is_dailybeast == 1].describe()

print "Num rows in dailybeast_df: " + str(dailybeast_df.shape[0])

print dailybeast_df.head()

#print dailybeast_df.columns

print " "

dailybeast_train_cols = ['gender', 'is_dailybeast'] + pp_cols

#print dailybeast_df.dtypes

dailybeast_logit = sm.Logit(dailybeast_df['rate_of_suicide'], dailybeast_df[dailybeast_train_cols])

dailybeast_result = dailybeast_logit.fit(cov_type="HC3")

# use bfgs
# dailybeast_result = dailybeast_logit.fit(cov_type="HC3", method='bfgs')


print " "
print "Dailybeast logit (cov_type = 'HC3') targeting 'rate_of_suicide', trained on: "
print dailybeast_train_cols
print " "
print "Dailybeast Logit Results: "
print dailybeast_result.summary()

print "Dailybeast logit (cov_type = 'HC3') targeting 'funding': "
dailybeast_fund_train_cols = ['gender', 'is_dailybeast'] + pp_cols
dailybeast_fund_logit = sm.Logit(dailybeast_df['funding'], dailybeast_df[dailybeast_fund_train_cols])

dailybeast_fund_result = dailybeast_fund_logit.fit(cov_type="HC3")
print dailybeast_fund_result.summary()

print "Dailybeast ols (cov_type = 'HC3') targeting 'accuracy_journalism': "
dailybeast_acc_train_cols = ['gender', 'is_dailybeast'] + pp_cols
#dailybeast_acc_ols = sm.OLS(dailybeast_df['accuracy_journalism'], dailybeast_df[dailybeast_fund_train_cols])
dailybeast_acc_ols = smf.ols(formula="accuracy_journalism ~ C(pp_dem)*C(is_dailybeast) + C(pp_gop)*C(is_dailybeast) + C(gender)", data=dailybeast_df)

dailybeast_acc_result = dailybeast_acc_ols.fit(cov_type="HC3")
print dailybeast_acc_result.summary()


print "Setting up new df for nyt"

nyt_df = df[df.is_dailybeast != 1]

nyt_cols = ['confidence_weight', 'accuracy_journalism', 'gender', 'fda_reg', 'weight', 'is_nyt'] + pp_cols

nyt_df = nyt_df[nyt_cols]

nyt_df = nyt_df.apply(pd.to_numeric, errors="coerce")

print "nyt describe control group"
print nyt_df[nyt_df.is_nyt != 1].describe()

print "nyt describe experiment group"
print nyt_df[nyt_df.is_nyt == 1].describe()

print "Num rows in nyt_df: " + str(nyt_df.shape[0])

print nyt_df.head()

#print nyt_df.columns

print " "

nyt_train_cols = ['gender', 'is_nyt'] + pp_cols

#print nyt_df.dtypes

nyt_logit = sm.Logit(nyt_df['weight'], nyt_df[nyt_train_cols])

nyt_result = nyt_logit.fit(cov_type="HC3")

# use bfgs
# nyt_result = nyt_logit.fit(cov_type="HC3", method='bfgs')


print " "
print "nyt logit (cov_type = 'HC3') targeting 'weight', trained on: "
print nyt_train_cols
print " "
print "nyt Logit Results: "
print nyt_result.summary()

print "nyt logit (cov_type = 'HC3') targeting 'fda_reg': "
nyt_fund_train_cols = ['gender', 'is_nyt']+pp_cols
nyt_fund_logit = sm.Logit(nyt_df['fda_reg'], nyt_df[nyt_fund_train_cols])

nyt_fund_result = nyt_fund_logit.fit(cov_type="HC3")
print nyt_fund_result.summary()

print "nyt ols (cov_type = 'HC3') targeting 'accuracy_journalism' (kitchen sink model): "
nyt_acc_train_cols = ['gender', 'is_nyt'] + pp_cols
#nyt_acc_ols = sm.OLS(nyt_df['accuracy_journalism'], nyt_df[nyt_fund_train_cols])
nyt_acc_ols = smf.ols(formula="accuracy_journalism ~ pp_dem*is_nyt + pp_gop*is_nyt + gender*is_nyt", data=nyt_df)

nyt_acc_result = nyt_acc_ols.fit(cov_type="HC3")
print nyt_acc_result.summary()

print "nyt ols (cov_type = 'HC3') targeting 'accuracy_journalism' (constrained model): "
#nyt_acc_ols = sm.OLS(nyt_df['accuracy_journalism'], nyt_df[nyt_fund_train_cols])
nyt_acc_2_ols = smf.ols(formula="accuracy_journalism ~ pp_dem + pp_gop + gender + is_nyt", data=nyt_df)

nyt_acc_2_result = nyt_acc_2_ols.fit(cov_type="HC3")
print nyt_acc_2_result.summary()


