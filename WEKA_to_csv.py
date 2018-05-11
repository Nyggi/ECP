WEKA_MULTIPLE_HOUSEHOLDS = False

if WEKA_MULTIPLE_HOUSEHOLDS:
    households = 'multiple'
else:
    households = 'single'

for i in range(24):
    feature_list = []
    filepath = 'WEKA_features/best_features_from_WEKA_' + str(households) + '/BestFeatures' + str(i) + '.csv'
    with open(filepath, 'r') as f:
        for line in f:
            if line[10:12] != ' 0':
                if line[-1:] == '\n':
                    feature_list.append(line[27:-1])
                else:
                    feature_list.append(line[27:])

    open(filepath, 'w').close()

    with open(filepath, 'w') as f:
        for thing in feature_list:
            f.write(str(thing))

            if thing != feature_list[-1]:
                f.write(",")