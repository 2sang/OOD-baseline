import numpy as np
import sklearn.metrics as sk


def right_wrong_distinction(model, test_images, test_labels):

    softmax_all = model.predict(test_images)
    right_all, wrong_all = split_right_wrong(softmax_all, test_labels)

    (s_prob_all, kl_all, mean_all, var_all) = entropy_stats(softmax_all)
    (s_prob_right, kl_right, mean_right, var_right) = entropy_stats(right_all)
    (s_prob_wrong, kl_wrong, mean_wrong, var_wrong) = entropy_stats(wrong_all)

    correct_cases = np.equal(np.argmax(softmax_all, 1), test_labels)
    accuracy = 100 * np.mean(np.float32(correct_cases))
    err = 100 - accuracy

    def print_result():
        print("\n[MNIST SUCCESS DETECTION]")
        print('MNIST Error (%)| Prediction Prob (mean, std) | PProb Right\
                (mean, std) | PProb Wrong (mean, std):')
        print(err, '|', np.mean(s_prob_all), np.std(s_prob_all), '|', np.mean(s_prob_right), np.std(s_prob_right), '|', np.mean(s_prob_wrong), np.std(s_prob_wrong))

        print('Success base rate (%):', round(accuracy,2), "({}/{})".format(len(right_all), len(softmax_all)))
        print('KL[p||u]: Right/Wrong classification distinction')
        print_curve_info(kl_right, kl_wrong)

        print('Prediction Prob: Right/Wrong classification distinction')
        print_curve_info(s_prob_right, s_prob_wrong)

        print('\nError Detection')
        print('Error base rate (%):', round(err,2), "({}/{})".format(len(wrong_all), len(softmax_all)))
        print_curve_info(-kl_right, -kl_wrong, True)

        print('Prediction Prob: Right/Wrong classification distinction')
        print_curve_info(-s_prob_right, -s_prob_wrong, True)
        
    print_result()
    

def in_out_distinction(model, indist_images, outdist_images, outdist_id):
    
    softmax_indist = model.predict(indist_images)
    softmax_outdist = model.predict(outdist_images)
    
    (s_prob_in, kl_in, mean_in, var_in) = entropy_stats(softmax_indist)
    (s_prob_out, kl_out, mean_out, var_out) = entropy_stats(softmax_outdist)
    
    def print_result():
        print("\n[MNIST-{} anomaly detection]".format(outdist_id))
        print('In-dist max softmax distribution (mean, std):')
        print(np.mean(s_prob_in), np.std(s_prob_in))
        
        print('Out-of-dist max softmax distribution(mean, std):')
        print(np.mean(s_prob_out), np.std(s_prob_out))

        print('\nNormality Detection')
        print('Normality base rate (%):', round(100*indist_images.shape[0]/(
                    outdist_images.shape[0] + indist_images.shape[0]),2))
        print('KL[p||u]: Normality Detection')
        print_curve_info(kl_in, kl_out)

        print('Prediction Prob: Normality Detection')
        print_curve_info(s_prob_in, s_prob_out)

        print('\nAbnormality Detection')
        print('Abnormality base rate (%):', round(100*outdist_images.shape[0]/(
                    outdist_images.shape[0] + indist_images.shape[0]),2))
        print('KL[p||u]: Abnormality Detection')
        print_curve_info(-kl_in, -kl_out, True)

        print('Prediction Prob: Abnormality Detection')
        print_curve_info(-s_prob_in, -s_prob_out, True)

    print_result()
    
    
def split_right_wrong(softmax_all, label):
    mask_right = np.equal(np.argmax(softmax_all, axis=1), label)
    mask_wrong = np.not_equal(np.argmax(softmax_all, axis=1), label)
    right, wrong = softmax_all[mask_right], softmax_all[mask_wrong]
    return right, wrong
    

def entropy_from_distribution(p, axis):
    return np.log(10.) + np.sum(p * np.log(np.abs(p) + 1e-11), axis=1, keepdims=True)


def print_curve_info(safe, risky, inverse=False):
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    if inverse:
        labels[safe.shape[0]:] += 1
    else:
        labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))


def entropy_stats(softmax):
    s_prob = np.amax(softmax, axis=1, keepdims=True)
    kl_all = entropy_from_distribution(softmax, axis=1)
    mean_all, var_all = np.mean(kl_all), np.var(kl_all)
    return s_prob, kl_all, mean_all, var_all
