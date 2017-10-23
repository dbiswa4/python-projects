import numpy as np
#from numpy import genfromtxt

def read_scores():
    scores_all = np.genfromtxt("Scores_all.csv", delimiter=",", skip_header=1)
    scores_all[np.isnan(scores_all)] = 0
    scores_all_int = scores_all.astype(int)

    #Validation
    #print(scores_all_int[0,:])
    #print(scores_all_int[1, :])

    print(scores_all_int)
    np.save('scores.npy', scores_all_int)

if __name__ == '__main__':
    print("HW9 - Question 1")
    read_scores()
