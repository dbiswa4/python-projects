import ml_utils as utils
import knn


def test_euclidean_distance(dataset):
    row0 = dataset[0]
    print row0
    count = 0
    for row in dataset:
        distance = utils.euclidean_distance(row0, row)
        print 'Distance between row0 and ' + ' row' +str(count) + ' is : ' + str(distance)
        print('Distance between row0 and row{0} is  : {1}').format(count, distance)
        print 'Distance between row0 and row{0} is  : {1}\n'.format(count, distance)
        count +=1

def test_get_neighbors(train, test_row, num_neighbors):
    neighbors = knn.get_neighbors(train, test_row, num_neighbors)
    print 'Neighbours:'
    for neighbor in neighbors:
        print(neighbor)

def range_test(low, high):
    for i in range(low, high):
        print i



if __name__ == '__main__':
    print 'This is a test module'

    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    test_euclidean_distance(dataset)

    test_get_neighbors(dataset, dataset[0], 3)

    #Test Prediction
    prediction = knn.predict_classification(dataset, dataset[0], 3)
    print('\nExpected %d, Got %d.' % (dataset[0][-1], prediction))

    #Test minmax
    minmax = utils.dataset_minmax(dataset)
    print 'minmax:\n', minmax

    filedata = utils.load_csv('abalone1.csv')
    print 'filedata:\n', filedata

    range_test(2, 4)

    #test str_column_to_int(dataset, column)
    #print 'cat var to numerical : \n', utils.str_column_to_int(filedata, 0)[0]
    #print 'cat var to numerical : \n', utils.str_column_to_int(filedata, 0)[1]
    unique_col_val, filedata = utils.str_column_to_int(filedata, 0)
    print 'filedata - cat var converted to numerical:\n',filedata

    floatdata = utils.str_columns_to_float(filedata)
    print 'float data : \n', floatdata

    #Test minmax
    minmax = utils.dataset_minmax(floatdata)
    print 'minmax:\n', minmax

    #Test normalize_dataset(dataset, minmax)
    normalized_dataset = utils.normalize_dataset(floatdata, minmax)

    print 'normalized_dataset : \n', normalized_dataset

    #Test get_accuracy()
    test_set = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
    actuals = ['a', 'a', 'b']
    predictions = ['a', 'a', 'a']
    accuracy = utils.accuracy_metric(actuals, predictions)
    print('Accuracy: {0}').format(accuracy)

    # Test confusion matrix with integers
    actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,3,5]
    predicted = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1,0,5]
    unique, matrix = utils.confusion_matrix(actual, predicted)
    print(unique)
    print(matrix)
    utils.print_confusion_matrix(unique, matrix)