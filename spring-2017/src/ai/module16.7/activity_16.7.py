import pandas as pd
import numpy as np

data = [['Sarah','Blonde','Average','Light','No','Burn'],
['Dana','Blonde','Tall','Average','Yes','None'],
['Alex','Brown','Short','Average','Yes','None'],
['Annie','Blonde','Short','Average','No','Burn'],
['Emily','Red','Average','Heavy','No','Burn'],
['Pete','Brown','Tall','Heavy','No','None'],
['John','Brown','Average','Heavy','No','None'],
['Katie','Blonde','Short','Light','Yes','None']]

df = pd.DataFrame(data, columns= ['Name', 'Hair','Hight','Weight','Lotion','Result'])
df.set_index(['Name'], inplace=True)

print df,'\n'


def avg_disorder(df,feature,target):
    
    epsilon = 2.0**-10 
    features_values = df[feature].unique()
    class_values = df[target].unique()
    #print 'features_values: ', features_values
    #print 'class_values: ', class_values
    
    k = len(features_values)
    n_t = df.shape[0]
    #print 'n_t: ', n_t
    
    avg_disorder=0
    for i in range(k):
        
        #print features_values[i]
        df_i = df[df[feature] == features_values[i]] 
        #print df_i
        n_i = df_i.shape[0]
        #print 'n_i: ', n_i
        
        tmp = []
        for label in class_values:
            n_i_c = sum(df_i[target] == label)
            #print label, ' >  n_i_c: ', n_i_c
            
            correct_rate = float(n_i_c)/n_i + epsilon# epsilon is to avoid infinite logs
            #print 'correct_rate: ', correct_rate 
            tmp.append(-correct_rate*np.log2(correct_rate) )
        avg_disorder += ((float(n_i)/n_t) * sum(tmp))
    
    return round(avg_disorder,2)

for col in df.columns - ['Result']:
    print 'avg_disorder of {}: {}\n'.format(col,avg_disorder(df,col,'Result'))
