def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    TN, FN, TP, FP = 0, 0, 0, 0
    num_test = prediction.shape[0]
    
    for i in range(num_test):
        if ground_truth[i]:
            if prediction[i]: TP+=1
            else: FN+=1
        else:
            if prediction[i]: FP+=1
            else: TN+=1
    
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FN + TN + FP)
        f1 = 2 * precision * recall / (precision + recall)
    
    except Exception as e:
        print('Ошибка:', e, '\n')

    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    curr_accuracy = 0
    num_test = prediction.shape[0]
    for j in range(num_test):
        if prediction[j] == ground_truth[j]: 
            curr_accuracy += 1    
    accuracy = curr_accuracy / num_test
    
    return accuracy
