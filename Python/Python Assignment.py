
# coding: utf-8

# In[1]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

mean([1,2,3,4,6,7,8,88,8])


# In[5]:

def median(numbers):
    sortedNumbers = sorted(numbers)
    middleNumbers = int(len(numbers)/2)
    if len(numbers) % 2 == 0:
        return ((sortedNumbers[middleNumbers - 1] + sortedNumbers[middleNumbers]) / 2.0)
    else:
        return sortedNumbers[middleNumbers]


# In[11]:

median([1,2,3,4,6,7,8,88,8,8])


# In[1]:

def mode(numbers):
    count = {}
    for x in numbers:
        if x in count:
            count[x] += 1
        else: 
            count[x] = 1    
    
    k = max(count, key=count.get)
    
    print  count, k


# In[48]:

mode([1,1,2,2,2,2,3,3,3,8,8,8,8,8,8,8,8])


# In[7]:

def percentile(list, p):
    if 0 <= p <= 100 :
        list = sorted(list)
        listLength = len(list)
        k = int(round((listLength * p/100.00) + 0.5))
        return list[k-1]
    else:
        return "Your P is not between 0-100"


# In[3]:

percentile([1,2,3,4,6,7,8,88,8], 88)


# In[8]:

def quartile(list, q):
    list = sorted(list)
    if len(list) <= 3:
        print "The length of the list has to be greater than 3 to calculate quartile"
    else:
        listLength = len(list)
        if q == 1:
            firstQuartile = int(listLength*0.25)
            return list[firstQuartile]
        elif q == 2:
            secondQuartile = median(list)
            return secondQuartile

        elif q == 3:
            thirdQuartile = int(listLength*0.75)
            return list[thirdQuartile]
        else:
            print "Please enter q between 1-3"

    


# In[144]:

quartile([1,2,3,4,6,7,8,88,8,8], 4)


# In[70]:

def num_range(begNum, endNum):
    list = range(begNum, endNum+1)
    return list
    


# In[204]:

num_range(0,4)


# In[6]:

def iqr(list):
    list = sorted(list)
    firstQuartile = quartile(list, 1)
    thirdQuartile = quartile(list, 3)
    InterquartileRange = thirdQuartile - firstQuartile
    return InterquartileRange


# In[5]:

iqr([1,2,3,4,6,7,8,88,8,8])


# In[203]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength
    
        
        
    
    


# In[204]:

var_p([1,2,3,4,6,7,8,88,8,8])


# In[206]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength


# In[207]:

var_s([1,2,3,4,6,7,8,88,8,8])


# In[208]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_p(list):
    return var_p(list)**0.5


# In[209]:

std_p([1,2,3,4,6,7,8,88,8,8])


# In[210]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_s(list):
    return var_s(list)**0.5


# In[211]:

std_s([1,2,3,4,6,7,8,88,8,8])


# In[212]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_p(list):
    return var_p(list)**0.5

def cv_p(list):
    coefficient_variation = std_p(list) / mean(list);
    return coefficient_variation


# In[213]:

cv_p([1,2,3,4,6,7,8,88,8,8])


# In[214]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_s(list):
    return var_s(list)**0.5

def cv_s(list):
    coefficient_variation = std_s(list) / mean(list);
    return coefficient_variation


# In[215]:

cv_s([1,2,3,4,6,7,8,88,8,8])


# In[218]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_s(list):
    return var_s(list)**0.5

def skewness(list):
    n = len(list)
    n1 = float(n - 1)
    n2 = float(n - 2)
    n3 = float(n1*n2)
    firstPart = float(n/n3)
    Xbar = mean(list)
    Std = std_s(list)
    secondPart = []
    for x in list:
        secondPart.append(float(((x - Xbar)/Std)**3))
       
    return firstPart*sum(secondPart)   



# In[219]:

skewness([1,2,3,4,6,7,8,88,8,8])


# In[222]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_p(list):
    return var_p(list)**0.5


def z_score(list, number):
    
    u = mean(list)
    popStd = std_p(list)
    
    return (number-u)/popStd
    
    


# In[223]:

z_score([1,2,3,4,6,7,8,88,8,8], 9)


# In[224]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_p(list):
    return var_p(list)**0.5


def z_score(list, number):
    
    u = mean(list)
    popStd = std_p(list)
    
    return (number-u)/popStd


def outlier_z(list, x):
    u = mean(list)
    s = std_p(list)
    
    if (x-u)/s > abs(3):
        return str(x) + " is an outlier"
    else:
        return str(x) + " is not an outlier"
   
    
    


# In[228]:

outlier_z([1,1,2,3,100,300], 1)


# In[236]:

def iqr(list):
    listLength = len(list)
    firstQuartile = (list[int(listLength*0.25)])
    thirdQuartile = (list[int(listLength*0.75)])
    InterquartileRange = thirdQuartile - firstQuartile
    return InterquartileRange
    
def outlier_iqr(list):
    listLength = len(list)
    firstQuartile = (list[int(listLength*0.25)])
    thirdQuartile = (list[int(listLength*0.75)])
    IQR = iqr(list)
    lowerBound = firstQuartile - 1.5 * IQR
    upperBound = thirdQuartile + 1.5 * IQR
    
    outlier = []
    
    for x in list:
        if x > upperBound:
            outlier.append(x)
        if x < lowerBound:
            outlier.append(x)
           
        
    return outlier
            
    
   
            


# In[238]:

outlier_iqr([1,2,3,4,6,7,8,88,8,8, 100])


# In[225]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def cov_p(listx, listy):
    n = len(listx)
    xBar = mean(listx)
    yBar = mean(listy)
    
    x_xBar = []
    y_yBar = []
    for x in listx:
            x_xBar.append(x-xBar)
    for y in listy:
            y_yBar.append(y-yBar)
   
    productX_Y = [a*b for a,b in zip(x_xBar, y_yBar)]
    return sum(productX_Y)/n
    


# In[226]:

cov_p([1, 2, 3, 4], [1, 2, 3, 4])


# In[239]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def cov_s(listx, listy):
    n = len(listx) - 1
    xBar = mean(listx)
    yBar = mean(listy)
    
    x_xBar = []
    y_yBar = []
    for x in listx:
            x_xBar.append(x-xBar)
    for y in listy:
            y_yBar.append(y-yBar)
   
    productX_Y = [a*b for a,b in zip(x_xBar, y_yBar)]
    return sum(productX_Y)/n
    
    


# In[240]:

cov_s([1, 2, 3, 4], [1, 2, 3, 4])


# In[241]:


def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_p(list):
    return var_p(list)**0.5

def cov_p(listx, listy):
    n = len(listx)
    xBar = mean(listx)
    yBar = mean(listy)
    
    x_xBar = []
    y_yBar = []
    for x in listx:
            x_xBar.append(x-xBar)
    for y in listy:
            y_yBar.append(y-yBar)
   
    productX_Y = [a*b for a,b in zip(x_xBar, y_yBar)]
    return sum(productX_Y)/n

def r_pearson_p(listx, listy):
    popSTD_x = std_p(listx)
    popSTD_y = std_p(listy)
    popCov = cov_p(listx, listy)
    
    return popCov/(popSTD_x * popSTD_y)


# In[242]:

r_pearson_p([1,2,3,4], [2,3,4,10])


# In[243]:

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_s(list):
    return var_s(list)**0.5

def cov_s(listx, listy):
    n = len(listx) - 1
    xBar = mean(listx)
    yBar = mean(listy)
    
    x_xBar = []
    y_yBar = []
    for x in listx:
            x_xBar.append(x-xBar)
    for y in listy:
            y_yBar.append(y-yBar)
   
    productX_Y = [a*b for a,b in zip(x_xBar, y_yBar)]
    return sum(productX_Y)/n

def r_pearson_s(listx, listy):
    samSTD_x = std_s(listx)
    samSTD_y = std_s(listy)
    samCov = cov_s(listx, listy)
    
    return samCov/(samSTD_x * samSTD_y)


# In[244]:

r_pearson_s([1,2,3,4], [2,3,4,10])


# In[247]:

def factorial(number):
    numberList = range(1, number +1)
    count = 1
    for x in numberList:
        count *= x
                
    return count
        


# In[248]:

factorial(10)


# In[249]:


def factorial(number):
    numberList = range(1, number +1)
    count = 1
    for x in numberList:
        count *= x
                
    return count
    
def combination(N, n):
    Nnum = factorial(N)
    nnum = factorial(n)
    N_n = factorial(N-n)
        
    return Nnum/(N_n*nnum)


# In[250]:

combination(10, 5)


# In[251]:


def factorial(number):
    numberList = range(1, number +1)
    count = 1
    for x in numberList:
        count *= x
                
    return count

def permutation(N, n):
    Nnum = factorial(N)
    nnum = factorial(n)
    N_n = factorial(N-n)
    
    
    return Nnum/(N_n)


# In[252]:

permutation(10, 5)


# In[261]:

def factorial(number):      
    numberList = range(1, number +1)      
    count = 1
    for x in numberList:
        count *= x
    return count
    
def combination(N, n):
    Nnum = factorial(N)
    nnum = factorial(n)
    N_n = factorial(N-n)
        
    return Nnum/(N_n*nnum)

def binomial_dist(n, p, x, true):   
    comb = combination(n,x)    
    pdf = comb*(p**x)*(1-p)**(n-x)   
    listx = range(0, x +1)  
    k = 0
    if true == False:
        return pdf  
    else:
        for j in listx:         
            k += combination(n,j)*(p**j)*(1-p)**(n-j)
        
    return k
    


# In[262]:

binomial_dist(4, 0.6, 3, True)



# In[534]:

binomial_dist(4, 0.6, 1, False)


# In[13]:

def factorial(number):      
        numberList = range(1, number +1)      
        count = 1
        for x in numberList:
             count *= x
        return count
    

def poisson_dist(mu, x, true):    
    xFact = factorial(x)
    listx = range(0, x +1)
    k = 0
    
    if true == False:   
        return (2.71828**(-mu)*(mu**(x)))/xFact
    
    if true == True:
        for j in listx:           
            k += (2.71828**(-mu)*(mu**(j)))/factorial(j)
        
    return k
    
        


# In[10]:

poisson_dist(5,3,True)


# In[11]:

poisson_dist(5,3, False)


# In[263]:

def factorial(number):      
        numberList = range(1, number +1)      
        count = 1
        for x in numberList:
             count *= x
        return count
    
def combination(N, n):
    Nnum = factorial(N)
    nnum = factorial(n)
    N_n = factorial(N-n)
        
    return Nnum/(N_n*nnum)

def hypergeometric_dist(N, r, n, x, true):
    
    listx = range(0, x +1)
    l = 0
    
    if true == False:
        k = (combination(r, x)*combination(N-r, n-x))/1.0/(combination(N, n))  
        return k
    
    if true == True:
        for j in listx:
            l += (combination(r, j)*combination(N-r, n-j))/1.0/(combination(N, n))
        
    return l


# In[264]:

hypergeometric_dist(52, 13, 5, 2, True)


# In[61]:

hypergeometric_dist(52, 13, 5, 2, False)


# In[265]:

def uniform_dist(a, b, x, true):
    if true == False:
        if x > b:
            return 0
        elif x < a:
            return 0
        else:
            return 1.0/(b-a)
        
        
    if true == True:
        if x < a:
            return 0
        if x >= b:
            return 1
        else:
            return (x-a)/1.0/(b-a)
    


# In[266]:

uniform_dist(10, 20, 15, True)


# In[41]:

def exponential_dist(mu, x, true):
    if true == False:
        return 1.0/mu*(2.71828**(-x/1.0/mu))
    if true == True:
        return 1 - (2.71828**(-x/1.0/mu))
        
    


# In[47]:

exponential_dist(22, 25, True)


# In[269]:

import numpy as np
import random
import matplotlib.pyplot as plt
import copy

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_s(list):
    return var_s(list)**0.5


def test_clt_with_uniform_dist(n, t):
    list = np.random.uniform(low=0.0, high=1.0, size= 1000)
    
    stdList = []
    meanList = []
    new_list = copy.deepcopy(stdList)
    
    for x in range(t):
        randList = random.sample(list, n)
        stdList.append(std_s(randList)) 
        meanList.append(mean(randList))


    plt.hist(stdList, bins = 10, normed=True)
    plt.title("standard deviation")
    plt.show()
    
    plt.hist(meanList, bins = 10, normed=True)
    plt.title("mean")
    plt.show()
    
    return std_s(stdList), mean(meanList)


# In[301]:

test_clt_with_uniform_dist(5, 1000)


# In[ ]:




# In[304]:

from scipy import stats

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_s(list):
    return var_s(list)**0.5

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_p(list):
    return var_p(list)**0.5

def ci_mean(L, a, true):
    meanL = mean(L)
    tValue = stats.t.ppf(1-a/2, len(L)-1)
    std_sL = std_s(L)
    n = len(L)**0.5
    sampleError= tValue* (std_sL/n)
    
    zValue = stats.norm.ppf(1-a/2)
    popError = zValue * (std_p(L)/n)
    if 0 <= a <= 1:
        if true == False:
            return meanL - sampleError, meanL + sampleError
        if true == True:
            return meanL - popError, meanL + popError


# In[305]:

ci_mean([1,2,3,4,5,6,7,8,9,10], 0.05, False)


# In[306]:

ci_mean([1,2,3,4,5,6,7,8,9,10], 0.05, True)


# In[485]:

from scipy import stats

def ci_proportion(pbar, n, a):
    zValue = stats.norm.ppf(1-a/2)
    k = pbar-pbar**2
    popError = zValue * ((k/n)**0.5)
    return max(0, pbar-popError), pbar + popError


# In[486]:

ci_proportion(0.5, 100, 0.05)


# In[307]:

from scipy import stats

def mean(numbers):
    count = 0
    for x in numbers:
        count += x
    return float(count) / max(len(numbers), 1)

def var_s(list):
    listLength = len(list) - 1
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_s(list):
    return var_s(list)**0.5

def var_p(list):
    listLength = len(list)
    meanList = mean(list)
    difXandMean = []
    difXandMeanSquared = []
    for x in (list):
        difXandMean.append(abs(x - meanList))
    for y in difXandMean:
        difXandMeanSquared.append(y**2)
    return sum(difXandMeanSquared)/listLength

def std_p(list):
    return var_p(list)**0.5

def hypo_test_for_mean(sampleMean, hypothesizedMean, n, a, SD, true):
    
    
    if true == False:
        t_crit = stats.t.ppf(1-a/2, n-1)
        t_score = (sampleMean-hypothesizedMean)/(SD/n**0.5)
        
        if sampleMean > hypothesizedMean:
            p_value_2_Tails = 2*(1-stats.t.cdf(t_score, n-1))
            p_value_1_Tail = 1-stats.t.cdf(t_score, n-1)
            test = p_value_2_Tails > a

        else:
            p_value_2_Tails = 2*(stats.t.cdf(t_score, n-1))
            p_value_1_Tail = stats.t.cdf(t_score, n-1)
            test = p_value_2_Tails > a
        return p_value_2_Tails, p_value_1_Tail, t_crit, t_score, test
    
    if true == True:
        z_crit = stats.norm.ppf(1-a/2, n-1)
        z_score = (sampleMean-hypothesizedMean)/(SD/n**0.5)
        
        if sampleMean > hypothesizedMean:
            p_value_2_Tails = 2*(1-stats.norm.cdf(sampleMean, hypothesizedMean, SD/(n**0.5)))
            p_value_1_Tail = (1-stats.norm.cdf(sampleMean, hypothesizedMean, SD/(n**0.5)))
            test = p_value_2_Tails > a

        else:
            p_value_2_Tails = 2*(stats.norm.cdf(sampleMean, hypothesizedMean, SD/(n**0.5)))
            p_value_1_Tail = (stats.norm.cdf(sampleMean, hypothesizedMean, SD/(n**0.5)))
            test = p_value_2_Tails > a
        return p_value_2_Tails, p_value_1_Tail, z_crit, z_score, test
 
            
    


# In[308]:

hypo_test_for_mean(36, 35, 34, 0.05, 7.858, False)


# In[313]:

from scipy import stats

def hypo_test_for_proportion(sampleProp, hypothesizedProp, n, a, tail):
    zcrit = stats.norm.ppf(1-a/2)
    k = ((hypothesizedProp-hypothesizedProp**2)/n)
    testStatistic = (sampleProp-hypothesizedProp)/k**0.5
    if tail == 0:
        if sampleProp < hypothesizedProp:
            p_value_2_tails = 2*(stats.norm.cdf(testStatistic))
            test = p_value_2_tails > a
        else:
            p_value_2_tails = 2*(1- stats.norm.cdf(testStatistic))
            test = p_value_2_tails > a
        return p_value_2_tails, zcrit, testStatistic, test
    if tail == 1:
        if sampleProp > hypothesizedProp:
            P_value_1_Tail = (1-stats.norm.cdf(testStatistic))
            test = P_value_1_Tail > a
            return P_value_1_Tail, zcrit, testStatistic, test
        else:
            return "no evidence"
    if tail == -1:
        if sampleProp < hypothesizedProp:
            P_value_1_Tail = stats.norm.cdf(testStatistic)
            test = P_value_1_Tail > a
            return P_value_1_Tail, zcrit, testStatistic, test
        else:
            return "no evidence"


# In[317]:

hypo_test_for_proportion(.795, .75, 44, 0.05, 1)


# In[318]:

hypo_test_for_proportion(.795, .75, 44, 0.05, 0)


# In[319]:

from scipy import stats

def power_in_hypo_test_for_mean(xbar, hypothesiezedU, n, sd, a, tail, true):
    testStatistic = (xbar-hypothesiezedU)/(sd/(n**0.5))
    if true == True:
        if tail == 0:
            type2 = 2*(1-stats.norm.cdf(testStatistic))
            return type2 
        if tail == 1:
            type2 = 1-stats.norm.cdf(testStatistic)
            return type2 
        if tail == -1:
            type2 = stats.norm.cdf(testStatistic)
            return type2 
    if true == False:
        if tail == 0:
            type2 = 2*(1-stats.t.cdf(testStatistic, n-1))
            return type2
        if tail == 1:
            type2 = 1-stats.t.cdf(testStatistic, n-1)
            return type2
        if tail == -1:
            type2 = stats.t.cdf(testStatistic, n-1)
            return type2


# In[320]:

power_in_hypo_test_for_mean(116.71, 112, 36, 12, 0.05, 1, False)


# In[130]:

from scipy import stats


def ci_for_mean_difference(u1, u2, n1, n2, sd1, sd2, a, true):
    if true == True:
        top = ((sd1**2)/n1 + (sd2**2)/n2)**2
        bottom = ((sd1**2)/n1)**2/(n1-1)
        bottom1 =((sd2**2)/n2)**2/(n2-1)
        df = top/(bottom+bottom1)
        
        j = (u1 - u2)
        k = ((sd1**2.0)/n1) + ((sd2**2.0)/n2)
        tcrit = stats.t.ppf(1-a/2, df)
        errorTerm = tcrit*(k**0.5)
        
        return j-errorTerm, j+errorTerm
    else:
        j = (u1 - u2)
        k = ((sd1**2.0)/n1) + ((sd2**2.0)/n2)
        zcrit = stats.norm.ppf(1-a/2)
        errorTerm = zcrit*(k**0.5)
        
        return j-errorTerm, j+errorTerm


# In[133]:

ci_for_mean_difference(9.31, 7.4, 100, 100, 4.67, 4.04, 0.05, True)


# In[321]:

from scipy import stats

def hypo_test_for_mean_difference(u1, u2, n1, n2, sd1, sd2, a, d, tail, true):
    
    if true == True:
        
        top = ((sd1**2)/n1 + (sd2**2)/n2)**2
        bottom = ((sd1**2)/n1)**2/(n1-1)
        bottom1 =((sd2**2)/n2)**2/(n2-1)
        df = top/(bottom+bottom1)
        k = ((sd1**2.0)/n1) + ((sd2**2.0)/n2)
        testStatistic = ((u1 - u2) - d)/k**0.5
        
        if tail == 0:
            tcrit = stats.t.ppf(1-a/2, df)
            p_value_2_tails = 2*(1-stats.t.cdf(testStatistic, df))
            test = p_value_2_tails > a
            
            return p_value_2_tails, testStatistic, tcrit, test
        if tail == 1:
            tcrit = stats.t.ppf(1-a, df)
            P_value_1_Tail = (1-stats.t.cdf(testStatistic, df))
            test = P_value_1_Tail > a
            return P_value_1_Tail, testStatistic, tcrit, test
        if tail == -1:
            tcrit = stats.t.ppf(a, df)
            P_value_1_Tail = stats.t.cdf(testStatistic, df)
            test = P_value_1_Tail > a
            return P_value_1_Tail, testStatistic, tcrit, test
    
    if true == False:
        
        top = (u1 - u2) - d
        bottom = (((sd1**2)/n1) + ((sd2**2)/n2))**0.5
        testStatistic = top/bottom
        if tail == 0:
            zcrit = stats.norm.ppf(1-a/2)
            p_value_2_tails = 2*(1- stats.norm.cdf(abs(testStatistic)))
            test = p_value_2_tails > a
            return p_value_2_tails, testStatistic, zcrit, test
        if tail == 1:
            zcrit = stats.norm.ppf(1-a)
            p_value_1_tail = (1 - stats.norm.cdf(testStatistic))
            test = p_value_1_tail > a
            return p_value_1_tail, testStatistic, zcrit, test
        if tail == -1:
            zcrit = stats.norm.ppf(a)
            p_value_1_tail = (stats.norm.cdf(testStatistic))
            test = p_value_1_tail > a
            return p_value_1_tail, testStatistic, zcrit, test


# In[322]:

hypo_test_for_mean_difference(9.31, 7.4, 100, 100, 4.67, 4.04, 0.05, 0, -1, False)


# In[323]:

hypo_test_for_mean_difference(9.31, 7.4, 100, 100, 4.67, 4.04, 0.05, 0, 0, True)


# In[314]:

from scipy import stats

def ci_for_proportion_difference(pbar1, pbar2, n1, n2, a):
    
    zValue = stats.norm.ppf(1-a/2)
    j = pbar1 - pbar2
    k = (((pbar1*(1-pbar1))/n1) + ((pbar2*(1-pbar2))/n2))**0.5
    popError = zValue * k
    return j-popError, j+popError


# In[315]:

ci_for_proportion_difference(0.3, 0.2, 100, 100,0.05)


# In[342]:

from scipy import stats

def hypo_test_for_proportion_difference(pbar1, pbar2, n1, n2, a, tail):
    pbar1 = pbar1/1.0/n1
    pbar2 = pbar2/1.0/n2
    pooled = ((pbar1*n1)+(pbar2*n2))/(n1+n2)
    testStatistic = (pbar1-pbar2)/(((pooled*(1-pooled)) * ((1.0/n1) + (1.0/n2)))**0.5)
    if tail == 0:
        zcrit = stats.norm.ppf(1-a/2)
        p_value_2_tails = 2*(1- stats.norm.cdf(abs(testStatistic)))
        test = p_value_2_tails > a
        return p_value_2_tails, testStatistic, zcrit, test
    if tail == 1:
        zcrit = stats.norm.ppf(1-a)
        p_value_1_tail = (1 - stats.norm.cdf(testStatistic))
        test = p_value_1_tail > a
        return p_value_1_tail, testStatistic, zcrit, test
    if tail == -1:
        zcrit = stats.norm.ppf(a)
        p_value_1_tail = (stats.norm.cdf(testStatistic))
        test = p_value_1_tail > a
        return p_value_1_tail, testStatistic, zcrit, test


# In[343]:

hypo_test_for_proportion_difference(30, 20, 100, 100, .05, -1)


# In[328]:

from scipy import stats

def ci_for_population_var(v, n, a):
    s = v**0.5
    chi_squared1 = stats.chi2.ppf(1-a/2, n-1)
    chi_squared2 = stats.chi2.ppf(a/2, n-1)

    E1 = ((n-1)*(s**2))/chi_squared1
    E2 = ((n-1)*(s**2))/chi_squared2
    return E1, E2


# In[329]:

ci_for_population_var(0.0025, 20, 0.05)


# In[339]:

from scipy import stats

def hypo_test_for_population_var(v, hypothesizedV, n, a, tail):
    s = v**0.5
    S = hypothesizedV**0.5
    testStatistic = ((n-1)* s**2)/S**2
    
    if tail == 1:
        chi_crit = stats.chi2.ppf(1-a, n-1)
        p_value_1_tail = (1 - stats.chi2.cdf(testStatistic, n-1))
        test = p_value_1_tail > a
        return p_value_1_tail, testStatistic, chi_crit, test
    if tail == -1:
        chi_crit = stats.chi2.ppf(1-a, n-1)
        p_value_1_tail = (stats.chi2.cdf(testStatistic, n-1))
        test = p_value_1_tail > a
        return p_value_1_tail, testStatistic, chi_crit, test


# In[341]:

hypo_test_for_population_var(4.9, 5, 24, 0.05, 1)


# In[335]:

from scipy import stats

def hypo_test_for_two_population_var(v1, v2, n1, n2, a, tail):
    s1 = v1**0.5
    s2 = v2**0.5
    testStatistic = (s1**2)/(s2**2)
    if tail == 1:
        f_crit = stats.f.ppf(1-a, n1-1, n2-1)
        p_value_1_tail = (1 - stats.f.cdf(testStatistic, n1-1, n2-1))
        test = p_value_1_tail > a
        return p_value_1_tail, testStatistic, f_crit, test
    if tail == -1:
        f_crit = stats.f.ppf(1-a, n1-1, n2-1)
        p_value_1_tail = (stats.f.cdf(testStatistic, n1-1, n2-1))
        test = p_value_1_tail > a
        return p_value_1_tail, testStatistic, f_crit, test


# In[336]:

hypo_test_for_two_population_var(48,20,26,16,0.05, 1)


# In[337]:

hypo_test_for_two_population_var(48,20,26,16,0.05, -1)

