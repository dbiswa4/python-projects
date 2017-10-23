import numpy as np


def group_avg(file_name='scores.npy'):
    np.set_printoptions(suppress=True)

    scores = np.load(file_name)

    m, n = scores.shape
    print(m, n)
    hw_avg = np.sum(scores[1:, 1:7])/float(44*6)
    print("Homewrok Average", hw_avg)
    print("Lab Average", np.sum(scores[1:, 7:18])/float(44*11))
    print("Project Average", np.sum(scores[1:, 18])/float(44*1))
    print("Midterm 1 Average", np.sum(scores[1:, 19])/float(44*1))
    print("Quiz Average", np.sum(scores[1:, 20:29])/float(44*9))
    print("Miterm 2 Average", np.sum(scores[1:, 29])/float(44*1))


if __name__ == '__main__':
    print("HW9 - Question 5")
    group_avg()
