import numpy as np
import numpy.ma as ma
import sklearn.metrics as sk


def right_wrong_distinction(model, test_images, test_labels):
    
    softmax_all = model.predict(test_images)
    right_all, wrong_all = split_right_wrong(softmax_all, test_labels)
    
    # Will be used in print_result()
    (s_prob_all, kl_all, mean_all, var_all) = entropy_stats(softmax_all)
    (s_prob_right, kl_right, mean_right, var_right) = entropy_stats(right_all)
    (s_prob_wrong, kl_wrong, mean_wrong, var_wrong) = entropy_stats(wrong_all)
    
    accuracy = 100*np.mean(np.float32(np.equal(np.argmax(softmax_all, 1), test_labels)))
    err = 100 - accuracy

    # Printing functions are taken from original repository without modifying.
    def print_result():
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
        safe, risky = -s_prob_right, -s_prob_wrong
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[safe.shape[0]:] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))
        
    print_result()
    

def in_out_distribution_distinction(model, outdist_string, indist_images, outdist_images):
    
    softmax_indist = model.predict(indist_images)
    softmax_outdist = model.predict(outdist_images)
    
    # Will be used in print_result()
    (s_prob_in, kl_in, mean_in, var_in) = entropy_stats(softmax_indist)
    (s_prob_out, kl_out, mean_out, var_out) = entropy_stats(softmax_outdist)
    
    def print_result():
        print('OOD Example Prediction Probability (mean, std):')
        print(np.mean(s_prob_out), np.std(s_prob_out))

        print('\nNormality Detection')
        print('Normality base rate (%):', round(100*indist_images.shape[0]/(
                    outdist_images.shape[0] + indist_images.shape[0]),2))
        print('KL[p||u]: Normality Detection')
        safe, risky = kl_in, kl_out
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[:safe.shape[0]] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

        print('Prediction Prob: Normality Detection')
        safe, risky = s_prob_in, s_prob_out
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[:safe.shape[0]] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

        # Fix starts here
        print('Normality base rate (%):', round(100*(1 - err/100)*in_examples.shape[0]/
            (out_examples.shape[0] + (1 - err/100)*in_examples.shape[0]),2))
        print('KL[p||u]: Normality Detection (relative to correct examples)')
        safe, risky = kl_r, kl_oos
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[:safe.shape[0]] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

        print('Prediction Prob: Normality Detection (relative to correct examples)')
        safe, risky = s_rp, s_p_oos
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[:safe.shape[0]] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))


        print('\n\nAbnormality Detection')
        print('Abnormality base rate (%):', round(100*out_examples.shape[0]/(
                    out_examples.shape[0] + in_examples.shape[0]),2))
        print('KL[p||u]: Abnormality Detection')
        safe, risky = -kl_a, -kl_oos
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[safe.shape[0]:] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

        print('Prediction Prob: Abnormality Detection')
        safe, risky = -s_p, -s_p_oos
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[safe.shape[0]:] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

        print('Abnormality base rate (%):', round(100*out_examples.shape[0]/
            (out_examples.shape[0] + (1 - err/100)*in_examples.shape[0]),2))
        print('KL[p||u]: Abnormality Detection (relative to correct examples)')
        safe, risky = -kl_r, -kl_oos
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[safe.shape[0]:] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

        print('Prediction Prob: Abnormality Detection (relative to correct examples)')
        safe, risky = -s_rp, -s_p_oos
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        labels[safe.shape[0]:] += 1
        examples = np.squeeze(np.vstack((safe, risky)))
        print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
        print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))
    print_result()
    
    
def split_right_wrong(softmax_all, label):
    mask_right = np.equal(np.argmax(softmax_all, axis=1), label)
    mask_wrong = np.not_equal(np.argmax(softmax_all, axis=1), label)
    right = softmax_all[mask_right]
    wrong = softmax_all[mask_wrong]
    return right, wrong
    

def entropy_in_softmax(softmax, axis):
    return np.log(10.) + np.sum(softmax * np.log(np.abs(softmax) + 1e-11), axis=1, keepdims=True)


def print_curve_info(safe, risky):
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[safe.shape[0]:] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))


def entropy_stats(softmax):
    s_prob = np.amax(softmax, axis=1, keepdims=True)
    kl_all = entropy_in_softmax(softmax, axis=1)
    mean_all, var_all = np.mean(kl_all), np.var(kl_all)
    return s_prob, kl_all, mean_all, var_all
