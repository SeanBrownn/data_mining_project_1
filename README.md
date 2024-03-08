Here we will describe how to run the codes in order to reproduce our results

Each of the files serves a unique purpose. We will explain here what each file does, from most to least important (meaning the most important ones are those that can be run to produce our results in the terminal)

cross_validation.py is where our performance measurements were drawn from. The data has already been loaded into that file, assuming it is placed in the same folder as 'cross_validation.py'. In order to reproduce our results for each of the models and each of the datasets, simply running the code should suffice (in VSCode, which is what we used, this can be done by simply pressing the "run" button for this file)

Some of the hyperparameter tuning was done in param_tuning.py. This file tests a lot of different hyperparameter combinations, so it's likely that it will take a while to run. It will also plot the distributions of accuracies for some of the models, which was used to assess the sensitivity of the models to hyperparameter tuning. The hyperparameter tuning for knn, decision tree, and AdaBoost were done in nearest_neighbor.py, decision_tree.py, and ada_boost.py respectively. The tuning methods are optimal_k(), optimal_depth(), and optimal_k() respectively.

For each of the different models we built, there is a serparate file with the corresponding name. These files can be run, but any output will just be our own testing. We conducted the proper performance testing on all of these algorithms in the cross_validation file.

Finally, the main.py class just contains some initial preprocessing, and doesn't need to be run. It just needs to be in the same folder as the rest of the code and everything will run fine.
