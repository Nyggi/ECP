feature_list = []

with open('BestFeatures.csv', 'r') as f:
    for line in f:
        feature = False
        string = ""

        for char in line:
            if feature == True:
                if char != "\n":
                    string = string + char

            elif char == "_":
                feature = True

        feature_list.append(int(string))

open('BestFeatures.csv', 'w').close()

with open('BestFeatures.csv', 'w') as f:
    for thing in feature_list:
        f.write(str(thing))

        if thing != feature_list[-1]:
            f.write(",")