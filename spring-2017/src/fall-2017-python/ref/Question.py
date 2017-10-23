#Question- Assignment 9
#This program will evaluate the final grade, average score per assignment, average score per assignment group.
#This will calculate depending upon the input data provided.

#This method will get us the letter Grade. Input will be score.

import numpy as np

AssignmentsName = ["HW 0","HW 1","HW 2","HW 3","HW 4","HW 5","Lab 01","Lab 02","Lab 03","Lab 04","Lab 05","Lab 06","Lab 07","Lab 08","Lab 09","Lab 10","Lab 11","Final Project","Midterm 1","Quiz 01","Quiz 02","Quiz 03","Quiz 05","Quiz 06","Quiz 07","Quiz 08","Quiz 09","Quiz 10","Midterm 2"]

LabsWeight = 0.30
HomeworkWeight = 0.20
QuizWeight = 0.15
Midterm1Weight = 0.10
Midterm2Weight = 0.15
FinalProjectWeight = 0.10

HomeworksGrades = (0, 5)
LabsGrades = (6, 16)
FinalProjectGrade = (17, 17)
Midterm1Grade = (18, 18)
QuizesGrade = (19, 27)
Midterm2Grade = (28, 28)

def getLetterGrade(score):
    grade = ""
    #lets round of to 2 digit to handle a case where use may enter 3 decimal point score.
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

def getStudentGrade(grades, totalWeights, weights):
    try:
        return np.sum((np.where(grades < 0, 0, grades) / totalWeights) * weights) * 100.0
    except Exception as ex:
        print("Error occured in getStudentGrade and the error is: ", ex)

def GetWeightForGroups(a, assignmentRange, assingmentWeight):
    try:
        assignmentRange = range(assignmentRange[0], assignmentRange[1]+1)
        n = len(assignmentRange)
        for i in assignmentRange:
            a[i] = assingmentWeight/n
    except Exception as ex:
        print("Error occured in GetWeightForGroups and the error is: ", ex)

def SetWeights(cols):
    #Setting up the weights to 29 columns because 1st one is student id.
    #Each homework weight should be 0.033
    #Each lab weight should come as 0.0272727272727
    weights = np.zeros(cols-1)
    GetWeightForGroups(weights, HomeworksGrades, HomeworkWeight)
    GetWeightForGroups(weights, LabsGrades, LabsWeight)
    GetWeightForGroups(weights, FinalProjectGrade, FinalProjectWeight)
    GetWeightForGroups(weights, Midterm1Grade, Midterm1Weight)
    GetWeightForGroups(weights, QuizesGrade, QuizWeight)
    GetWeightForGroups(weights, Midterm2Grade, Midterm2Weight)
    return weights

def GetAverageGrades(numericGrades):
    try:
        avgClass = np.empty((1,2), dtype='<U5')
        avgClassNumeric = numericGrades[:,[1]].sum() / len(numericGrades[:,[1]])
        avgClassLetterGrade = getLetterGrade(avgClassNumeric)
        avgClass[0] = (avgClassNumeric,avgClassLetterGrade)
        print("Class Average: {0:3.2f} Grade: {1}".format(avgClassNumeric,avgClassLetterGrade))
        np.save("AverageGrades.npy",avgClass)
    except Exception as ex:
        print("Error occured in GetAverageGrades and error is :",ex)

def GetAverageAssignment(inputData,cols):
    try:
        avgAssignment = np.zeros((cols-1, 2))
        avgAssignmentLabel = np.zeros((cols-1, 2), dtype='<U15')
        print("Averages for each assignment")
        for i in range(1,cols):
            assignment = inputData[1:,[i]]
            count = len(assignment)
            average = np.where(assignment < 0, 0, assignment).sum() / count
            print("{0} Average: {1:3.2f}".format(AssignmentsName[i-1], average))
            avgAssignment[i-1] = (i-1, average)
            avgAssignmentLabel[i-1] = (AssignmentsName[i-1], average)
        np.save("AverageAssig.npy", avgAssignmentLabel)
        return avgAssignment
    except Exception as ex:
        print("error is GetAverageAssignment and error is :", ex)

def GetAverageGroup(avgAssignment):
    print("Average for group assignment")
    try:
        assignmentGroupAverage = np.zeros((6, 2), dtype='<U15')
        homeworkAverage = avgAssignment[HomeworksGrades[0]:HomeworksGrades[1]+1,[1]].sum() / len(range(HomeworksGrades[0], HomeworksGrades[1]+1))
        print("Homework Average: {0:3.2f}".format(homeworkAverage))
        assignmentGroupAverage[0] = ("Homework", homeworkAverage)
        labsAverage = avgAssignment[LabsGrades[0]:LabsGrades[1]+1,[1]].sum() / len(range(LabsGrades[0], LabsGrades[1]+1))
        print("Labs Average: {0:3.2f}".format(labsAverage))
        assignmentGroupAverage[1] = ("Labs", labsAverage)
        finalProjectAverage = avgAssignment[FinalProjectGrade[0]:FinalProjectGrade[1]+1,[1]].sum() / len(range(FinalProjectGrade[0], FinalProjectGrade[1]+1))
        print("Final Project Average: {0:3.2f}".format(finalProjectAverage))
        assignmentGroupAverage[2] = ("Final Project", finalProjectAverage)
        midterm1Average = avgAssignment[Midterm1Grade[0]:Midterm1Grade[1]+1,[1]].sum() / len(range(Midterm1Grade[0], Midterm1Grade[1]+1))
        print("Midterm 1 Average: {0:3.2f}".format(midterm1Average))
        assignmentGroupAverage[3] = ("Midterm 1", midterm1Average)
        quizAverage = avgAssignment[QuizesGrade[0]:QuizesGrade[1]+1,[1]].sum() / len(range(QuizesGrade[0], QuizesGrade[1]+1))
        print("Quiz Average: {0:3.2f}".format(quizAverage))
        assignmentGroupAverage[4] = ("Midterm 1", quizAverage)
        midterm2Average = avgAssignment[Midterm2Grade[0]:Midterm2Grade[1]+1,[1]].sum() / len(range(Midterm2Grade[0], Midterm2Grade[1]+1))
        print("Midterm 2 Average: {0:3.2f}".format(midterm2Average))
        assignmentGroupAverage[5] = ("Midterm 2", midterm2Average)
        np.save("AverageGroup.npy", assignmentGroupAverage)
    except Exception as ex:
        print("Error occured in GetAverageGroup and error is :", ex)

def GetGradeForEachStudent(inputData,rows,totalWeights,weights):
    try:
        numericGrades = np.zeros((rows-1,2))
        letterGrades = np.empty((rows-1,2), dtype='<U5')
        for r in range(1, rows):
            studentId = inputData[r,0]#fix the column as 0th column is the student ID
            numericGrades[r-1] = (studentId, getStudentGrade(inputData[r][1:], totalWeights, weights))
            letterGrades[r-1] = (studentId, getLetterGrade(numericGrades[r-1,1]))
        studentGrades = np.hstack((letterGrades[:,[0]], numericGrades[:,[1]], letterGrades[:,[1]]))
        print("Grades for each student")
        for sg in studentGrades:
            print("StudentId: {0} Average: {1:3.2f} Grade: {2}".format(sg[0], float(sg[1]), sg[2]))        
        np.save("FinalGrades.npy", studentGrades)
        return numericGrades
    except Exception as ex:
        print("Error occured in GetGradeForEachStudent and error is :", ex)

def main():
    #Reading the data.
    try:
        inputData = np.load("C:\\Users\\himanshu\\Downloads\\Grades.npy")
        #inputData = np.load("Grades.npy")
        #Get the rows and cols.
        rows, cols = inputData.shape
        #The first row of inputData is Weights of individual assignment. This will be used in computiing the %
        totalWeights = inputData[0][1:]
        weights = SetWeights(cols)
        #Create 2-d arrays for students. One with ID and score and other with ID and letter grade.
        numericGrades = GetGradeForEachStudent(inputData,rows,totalWeights,weights)
        GetAverageGrades(numericGrades)
        avgAssignment = GetAverageAssignment(inputData,cols)
        GetAverageGroup(avgAssignment)
    except Exception as ex:
        print("Error occured in main function and error is :",ex)
       
main()