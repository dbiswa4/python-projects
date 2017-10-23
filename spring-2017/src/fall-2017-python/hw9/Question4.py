import numpy as np

assignments_name = ["HW 0","HW 1","HW 2","HW 3","HW 4","HW 5","Lab 01","Lab 02","Lab 03","Lab 04","Lab 05","Lab 06","Lab 07","Lab 08","Lab 09","Lab 10","Lab 11","Final Project","Midterm 1","Quiz 01","Quiz 02","Quiz 03","Quiz 05","Quiz 06","Quiz 07","Quiz 08","Quiz 09","Quiz 10","Midterm 2"]


def calcAvg(scores):
    sum = np.sum(scores[1:])
    l = len(scores)
    avg = sum/float(l)
    return avg


def avg_scores(file_name='scores.npy'):
    np.set_printoptions(suppress=True)

    scores = np.load(file_name)

    m, n = scores.shape
    print(m, n)

    for i in range(1, 30):
        score = scores[:, i]
        avg = calcAvg(score)
        print(assignments_name[i-1], "average : ", round(avg, 2))

if __name__ == '__main__':
    print("HW9 - Question 4")
    avg_scores()
