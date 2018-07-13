import numpy as np
import numpy.ma as ma
import sklearn.metrics as sk


def right_wrong_distinction(model, test_images, test_labels):
    softmax_all = np.array(model.predict(test_images))
    right_all, wrong_all = split_right_wrong(softmax_all, test_labels)
    
    (s_prob_all, kl_all, mean_all, var_all) = entropy_stats(softmax_all)
    (s_prob_right, kl_right, mean_right, var_right) = entropy_stats(right_all)
    (s_prob_wrong, kl_wrong, mean_wrong, var_wrong) = entropy_stats(wrong_all)
    accuracy = 100*np.mean(np.float32(np.equal(np.argmax(softmax_all, 1), test_labels)))
    err = 100 - accuracy

    print('MNIST Error (%)| Prediction Prob (mean, std) | PProb Right (mean, std) | PProb Wrong (mean, std):')
    print(err, '|', np.mean(s_prob_all), np.std(s_prob_all), '|', np.mean(s_prob_right), np.std(s_prob_right), '|', np.mean(s_prob_wrong), np.std(s_prob_wrong))

    print('\nSuccess Detection')
    print('Success base rate (%):', round(accuracy,2))
    print('KL[p||u]: Right/Wrong classification distinction')
    safe, risky = kl_right, kl_wrong
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

    print('Prediction Prob: Right/Wrong classification distinction')
    safe, risky = s_prob_right, s_prob_wrong
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    #print("graph:")
    #fpr, tpr, thr = sk.roc_curve(labels, examples)
    #plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

    print('\nError Detection')
    print('Error base rate (%):', round(err,2))
    safe, risky = -kl_right, -kl_wrong
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[safe.shape[0]:] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    print('KL[p||u]: Right/Wrong classification distinction')
    print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

    print('Prediction Prob: Right/Wrong classification distinction')
    safe, risky = -s_prob_right, -s_prob_right
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[safe.shape[0]:] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

def in_out_distribution_distinction(model, test_images, test_labels):
    
    
def split_right_wrong(softmax_all, label):
    mask_right = np.equal(np.argmax(softmax_all, axis=1), label)
    mask_wrong = np.not_equal(np.argmax(softmax_all, axis=1), label)
    right = softmax_all[mask_right]
    wrong = softmax_all[mask_wrong]
    return right, wrong
    

def entropy_in_softmax(softmax, axis):
    return np.log(10.) + np.sum(softmax * np.log(np.abs(softmax) + 1e-11), axis=1, keepdims=True)


def entropy_stats(softmax):
    s_prob = np.amax(softmax, axis=1, keepdims=True)
    kl_all = entropy_in_softmax(softmax, axis=1)
    mean_all, var_all = np.mean(kl_all), np.var(kl_all)
    return s_prob, kl_all, mean_all, var_all
