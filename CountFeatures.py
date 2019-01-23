import pickle
if __name__ == '__main__':
    w = pickle.load(open('w_pickle/w_simple_100', 'rb'))
    res = {}
    for i in range(1, 15):
        res['f{}'.format(i)] = 0
    for key in w.keys():
        key_format = key.split('_')
        if key_format[0] == 'f':
            kf = key_format[0] + key_format[1]
        else:
            kf = key_format[0]
        res[kf] += 1
    print(res)
