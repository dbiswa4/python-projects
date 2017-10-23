import numpy as np

HomeworksGrades = (0, 5)
LabsGrades = (6, 16)
FinalProjectGrade = (17, 17)
Midterm1Grade = (18, 18)
QuizesGrade = (19, 27)
Midterm2Grade = (28, 28)

def group_avg(file_name='scores.npy'):
    np.set_printoptions(suppress=True)

    scores = np.load(file_name)

    m, n = scores.shape
    print(m, n)

if __name__ == '__main__':
    print("HW9 - Question 5")
    group_avg()
