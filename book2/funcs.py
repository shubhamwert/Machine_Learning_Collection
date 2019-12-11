import math
import numpy as np
def pow_transformation(y: np.array,lamda):
    
    if(lamda!=0):
        yw=np.power(y,lamda)-1
        yw=yw/lamda
    else:
        yw=np.log(y)
    return yw
def bayes_probility_binary(probability_of_eventB,probility_of_eventC,probability_of_eventA_givenB,probability_of_eventA_givenC):
    p=probability_of_eventA_givenB*probability_of_eventB
    p=p/total_probability(p,probability_of_eventA_givenC*probility_of_eventC)
    return p

def total_probability(p1,p2):
    return p1+p2


def probability(favourable_cases,total_cases):
    return favourable_cases/total_cases

def test_bayes():
    print(bayes_probility_binary(0.5,0.5,0.9,0.5))
def laplace_smoothing(favourable_cases,total_cases,total_word,lamda=1):

    return (favourable_cases+lamda)/(total_cases+lamda*total_word)

def prob_against_odds(in_favour,in_odds):
    return in_favour/(in_favour+in_odds)
