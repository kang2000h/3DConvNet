import numpy as np
import pandas as pd


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

# update each prediction count per each update trial
def log_error_board(filepath, additional_p_id, additional_label, additional_pred, num_classes):
    print("[!] Start Logging Error into Error Board")
    additional_p_id = np.array(additional_p_id)
    #additional_label = np.argmax(np.array(additional_label), axis=1)  # make it non-one-hot shape
    additional_label = np.array(additional_label)
    additional_pred = np.array(additional_pred)

    print("additional_p_id shape", additional_p_id.shape)
    print("additional_label shape", additional_label.shape)
    print("additional_pred shape", additional_pred.shape)

    additional_pred = np.argmax(additional_pred, axis=1)
    additional_pred = get_one_hot(additional_pred, nb_classes=num_classes)
    additional_pred_t = np.transpose(additional_pred).astype(np.int32)
    #print("pred one hot", pred_t)

    try :
        #csv_p_label
        conf_matrix = pd.read_csv(filepath, index_col=0)
        print("[*] read present conf matrix in place")
        #print(conf_matrix)
    except FileNotFoundError as fnfe:
        conf_matrix = dict()

        conf_matrix['p_id'] = np.array(additional_p_id).tolist()
        conf_matrix['label'] = np.array(additional_label).tolist()


        for ind in range(num_classes):
            conf_matrix[str(ind)] = additional_pred_t[ind].tolist()

        data = pd.DataFrame(conf_matrix)
        print("[!] Create New Conf-Matrix")
        data.to_csv(filepath, mode="w")
        return

    origin_p_id = list(conf_matrix['p_id'])
    origin_label = list(conf_matrix['label'])

    origin_pred = [[] for _ in range(num_classes)] # [[0, 3, 0, 0], [3, 0, 3, 3]]
    for c_ind in range(num_classes):
        for origin_ind in range(len(origin_p_id)):
            origin_pred[c_ind].append(conf_matrix[str(c_ind)][origin_ind])

    for ind, a_p_id in enumerate(additional_p_id):
        try:
            overlapped_p_ind = origin_p_id.index(a_p_id)
            for c_ind in range(num_classes):
                if additional_pred_t[c_ind][ind] == 1:
                    origin_pred[c_ind][overlapped_p_ind] += 1

        # if there is no p_ind above, add p_ind as a new p_ind into origin_p_id
        except ValueError as ve:
            origin_p_id.append(a_p_id)
            origin_label.append(additional_label[ind])
            for c_ind in range(num_classes):
                origin_pred[c_ind].append(additional_pred_t[c_ind][ind])

    conf_matrix = dict() # need to be initialize to make a new DataFrame
    conf_matrix['p_id']=origin_p_id
    conf_matrix['label'] = origin_label
    for ind in range(num_classes):
        conf_matrix[str(ind)]=origin_pred[ind]

    # Calc Percentage of model with correction recognotion
    print("[*] calc percentage...")
    percentage_list = []
    for ind, l in enumerate(conf_matrix['label']):
        sum = 0
        percentage = 0.0
        for cind in range(num_classes):
            sum += origin_pred[cind][ind]
        percentage_list.append((origin_pred[int(l)][ind] / sum) * 100)
    conf_matrix['percentage(%)'] = percentage_list

    # print("debug conf mat")
    # print(conf_matrix) # {'p_name': ['p1', 'p2', 'p3', 'p4'], 'label': [1, 0, 1, 0], '0': [0.0, 1.0, 0.0, 0.0], '1': [1.0, 0.0, 1.0, 1.0]}
    data = pd.DataFrame(conf_matrix)
    data.to_csv(filepath, mode="w")
