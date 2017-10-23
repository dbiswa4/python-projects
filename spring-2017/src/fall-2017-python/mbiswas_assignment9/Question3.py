import numpy as np
#from numpy import genfromtxt

def getLetterGrade(score):
    grade = ""
    score = round(score,2)
    if(score>=93 and score<=100):
        grade = "A"
    elif (score>=90 and score<=92.99):
        grade = "A-"
    elif (score>=86 and score<=89.99):
        grade = "B+"
    elif (score>83 and score<=85.99):
        grade = "B"
    elif (score>80 and score<=82.99):
        grade = "B-"
    elif (score>76 and score<=79.99):
        grade = "C+"
    elif (score>73 and score<=75.99):
        grade = "C"
    elif (score>70 and score<=72.99):
        grade = "C-"
    elif (score>66 and score<=69.99):
        grade = "D+"
    elif (score>60 and score<=65.99):
        grade = "D"
    elif (score>0 and score<=59.99):
        grade = "F"
    return  grade


def final_scores(file_name='score_headings.npy'):
    np.set_printoptions(suppress=True)

    scores = np.load(file_name)

    student_ids = scores[:, 0]
    #print("Student Ids : ", student_ids)

    total_scores_only = scores[:, 1:].sum(axis=1)
    total_scores_only_rounded = np.around(total_scores_only, decimals=1)

    ids_scores = np.column_stack((student_ids.astype(str), total_scores_only_rounded.astype(str)))

    vfunc_grade = np.vectorize(getLetterGrade)
    grades = vfunc_grade(total_scores_only_rounded)

    final_scores = np.column_stack((ids_scores, grades.astype(str)))

    print("Final Scores : ")
    print(final_scores)



if __name__ == '__main__':
    print("HW9 - Question 3")
    final_scores()
