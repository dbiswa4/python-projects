import numpy as np
#from numpy import genfromtxt

def score_headings(file_name='scores.npy'):
    scores = np.load(file_name)

    #data load validation
    #print(scores[0,:])
    #print(scores[1, :])

    np.set_printoptions(suppress=True)

    student_ids = scores[1:, 0]
    #print("Student Ids : ", student_ids)

    #1. Weighted homework scores
    # Homework weights is 20%

    # From 1 to rest of all rows
    # 6 columns (index=0 to index=6 colum).
    # Why 7? Upper bound excluded
    hw_total_weight = scores[:1, 1:7].sum(axis=1)[0]
    hw_totals = scores[1:, 1:7].sum(axis=1)
    hw_weighted_scores = (hw_totals/float(hw_total_weight))*20
    #print(hw_weighted_scores)

    #2. Weighted lab scores (index=7 to index=17 col)
    #Lab weights is 30%
    hw_lab_scores = (scores[1:, 7:18].sum(axis=1)/float(scores[:1, 7:18].sum(axis=1)[0]))*30
    #print("lab: ", hw_lab_scores)

    #3. Weighted final project scores (index=18 col)
    #Final project weight is 10%
    hw_finalproj_scores = (scores[1:, 18]/float(scores[0,18]))*10
    #print("Final Proj : ", hw_finalproj_scores)

    #4. Weighted midterm 1 scores (index=19)
    #Midterm 1 weight is 10%
    hw_mid1_scores = (scores[1:, 19]/float(scores[0,19]))*10
    #print("Midterm 1 : ", hw_mid1_scores)

    #5. Weighted Quiz scores (index=20 to index=28 col)
    #Quiz weight is 15%
    hw_quiz_scores = (scores[1:, 20:29].sum(axis=1)/float(scores[:1, 20:29].sum(axis=1)[0]))*15
    #print("Quiz : ", hw_quiz_scores)

    #4. Weighted midterm 2 scores (index=29)
    #Midterm 1 weight is 15%
    hw_mid2_scores = (scores[1:, 29]/float(scores[0,29]))*15
    #print("Midterm 2 : ", hw_mid2_scores)

    score_headings = np.column_stack((student_ids, hw_weighted_scores, hw_lab_scores, hw_finalproj_scores, hw_mid1_scores, hw_quiz_scores, hw_mid2_scores))

    np.around(score_headings, decimals = 1)

    print("Print Final result:")
    print(score_headings)
    np.save('score_headings.npy', score_headings)



if __name__ == '__main__':
    print("HW9 - Question 2")
    score_headings()
