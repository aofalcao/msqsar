# MS-QSAR 


[QSAR](https://en.wikipedia.org/wiki/Quantitative_structure%E2%80%93activity_relationship) stands for Quantitative Structure Activity Relationship and is typically used in contects of inslico modeling approaches for pharmacological activity modeling.  `MSQSAR`  uses chemical similarity for inference and analysis of the chemical space. It allows for the folowing QSAR operations

* Create QSAR models (quatitative or qualitative models)
* Evaluate data set modelability
* Validate models
* Screen datasets

Despite being implemented in 100% pure Python it was developed taking advantage of very efficient libraries for numerical processing, machine learning and data processing.


## MS-QSAR requirements

`MSQSAR` is essentially Python software that relies on a number of external libraries. Namely:

* numpy
* scikit-learn
* rdkit

Of those, rdkit is the trickiest to install. Anaconda makes the rdkit installation a breeze and this is the process recommended, although it might be possible to use a [docker container](https://hub.docker.com/search?q=rdkit&type=image) with rdkit already installed. To install rdkit with anaconda we can follow the [procedure detailed in the official rdkit distribution](https://www.rdkit.org/docs/Install.html):

First we create an environment `rdkit-env` like this:

```sh
$ conda create -c conda-forge -n rdkit-env rdkit
```

and then to activate we just activate the environment to start working

```sh
$ conda activate rdkit-env
```

Windows users can omit the `conda` and just use `activate rdkit-env` at the windows command line prompt


With Anaconda, numpy is installed by default and installing the others can be performed simply with: 

```sh
$ conda install scikit-learn
$ conda install requests
```

Just make sure these are installed within the `rdkit` environment. 

### Installing MSQSAR

This should be a trivial process. Just copy the `msqsar.py` files to a new folder in your woring directory: 

```
$ mkdir qsartools
$ cp [source code path]/*.py qsartools/.
$ cd qsartools
```

### SAR files

SAR files are the required format for most of the operations of this set of tools. SAR files are text files that should store activity data for a given target for a set of molecules and each line represents one molecule. Columns are separated by `<tab>`. The basic structur of a SAR file is

* The first column is an alphanumeric identifier for the molecule (should be unique);
* The second column contains the activity registered for that molecule (may be binary qualitative)
* The third column contains the molecule SMILES

Here is a sample SAR file, where the identifiers are the ChEMBL IDs

```
CHEMBL158973	0.5000	CN(C)CCSC(C)(C)C
CHEMBL476516	0.1880	NCCc1ccc(Br)cc1
CHEMBL309689	0.0000	Oc1noc2c1CCNCC2
...
CHEMBL2331792	0.0000	COc1noc2c1CCNCC2
CHEMBL2331804	0.0000	Cn1oc2c(c1=O)CCNCC2
CHEMBL2436555	0.2200	NC1=Nc2ccc(Cl)cc2CN1
```

A binary qualitative file may have the following structure

```sh
CHEMBL147434	A	COc1ccc2[nH]cc(C3CC3N)c2c1
CHEMBL148091	A	COc1cccc2[nH]cc(C3CC3N)c12
CHEMBL149564	NA	COc1ccc2[nH]cc(C(C)CN)c2c1
CHEMBL160610	N	NC1c2ccccc2Cc2ccccc21
CHEMBL161203	NA	CN(C)CCSc1nc(C(C)(C)C)cs1
```

Where 'A' stands for active, 'N', is a non-active molecule and 'NA' means undeterminate. Occasionally, cs-tools will produce SAR files with predicted values from models and 'NA' means that it was not possible to ascertain a value


# Using MSQSAR

## Calling for --help

Running `msqar` with the `--help` option will display the usage options of the tool

```
$ python msqsar.py --help
```

will give:

```
MS-QSAR - (C) Andre Falcao DI/FCUL version 0.2.20230605
Usage: This is a python tool for building Metric space QSAR models.
       This tool requires an enviroment where RDkit and scikit-learn is installed
       To run type python msqsar.py [method] -in [input .sar file] [options]

method can be:
    eval - evaluates the quality of a given data set for making inference
    test - checks predictions made against a data set
    infer - makes predictions against a data set - no stats are provided
    screen - screens a large database, identifying the most likely candidates (classification only)

Control parameters:
    -in file_name - the data set used for model building (required) (.sar format)
    -test file_name - data set required for method=test (.sar format)
    -scr filename - file with data for screening. (.smi format)
    -out filename - file where the output is stored. if ommited, redirects to stdout
    -fpR N - fingerprint radius (default: 2)
    -fpNB N - fingerprint number of bits (default: 1024)
    -max_dist f - maximum distance for prediction (default: 0.9)
    -min_sims N - minimum number of instances for modeling (default: 5)
    -max_sims - maximum number of instances for modeling (default: 20)
    -algo (SVM | RF)- machine learning algorithm  (default: RF)
    -ntrees - Number of trees in random forest (RF (default: 20)
    -nprocs - Number of processes if -parallel option enabled
    
Output Control Options:
    -silent - no intermediate output at all
    -detail - in the final output show the individual predictions
    -nostats - do not show end model statistics for methods eval and test
    -parallel - run the process in parallel (currently only for screening)
```


## Checking the modelability of a test file

This is the main procedure utility is the main one for modeling. It uses metric space modeling to try to infer the local structure of a dataset, and accordingly perform an evaluation of the coherency of such a space. A highly coherent space will preserve similar activity profiles in close chemical-space regions. As such, for each molecule in the data set, its activity will be inferred according to its closest neighbours, but not by using a k-NN algorithm, rather by defining a non-linear chemical property layer over the chemical space of the molecules   


### Modelability for regression data sets

Here is a simple run in a regression type problem to infer the activity of molecules to the Histamine 1 Receptor (H1R). This will require just that the user specify the file location of the .sar file that will be used for evaluation.


```
$ python msqsar.py eval -in data/H1.sar
```

The corresponding output will show which operation is being processed and in the end will output the regression statistics

```
Evaluating!
1. Reading molecules... Model Type: Regression
2. Determining structure...
3. Computing chemical space...
4. Finding modeling matches...
5. Fit local models... Done!
RMSE:  0.1456
Pearson:  0.8926
PVE:  0.7960
%Predicted:  0.887
```

Note that the program detected this as a regression problem and produced the appropriate regression statistics. Now what 1do they mean?
* RMSE - root mean squared error. is a measure of the differences between predicted and real value. For this type of data (pIC50, scaled between 0 and 1) values below 0.2 mean that the model is making reliable predictions, and values below 0.1 suggest extremely precise models.
* Pearson - this is the correlation between the predicted and the estimated values. It allows to evaluate the robustness of the model
* PVE - Percentage of Varaince explained. This is similar to the R2. It measures how much of the variance of the activity variable is being captured by the model

The last value $%predicted$ has a different meaning. As this approach is strongly anchored on strutuctural similarrity, if a molecule is deemed too distant from its closest neighbours then it cannot be predicted. This score thus means how many of the molecules of the dataset have been assignmed a value. In this case, about 11.3% of all molecules were not predicted (`0.113 = 1 - 0.887`). The maximum distance of the molecules used for prediction can be user defined thus allowing for the control of the size of the prediction space (modelable space)

### Modelability for Classification datasets

There are no special directives to fit a classification problem, other than the data should appear in the .SAR format with the activity of each molecule being one of A, N, or NA, for active, non-active and not assigned

In this example however we are using two option settings. The first one to control the maximum distance (`-max_dist`) we are allowing for modeling and a second one (`-silent`) to suppress the intermediate output and just produce the final statistics

```sh
$ python msqsar.py eval -in data/EGFR_class.sar -max_dist 0.7 -silent
```

The above produced the results below

```
MCC:  0.7545
F1:  0.8543
Precision:  0.8870
Recall:  0.8240
%Predicted:  0.962
```

These mean:
* MCC - [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient), is a robust statistic to acess the quality of a binary classifier
* F1 - is the widely used F1 score that corresponds to the geometric average of Precision and Recall. It is particulalrly useful when the number of actives is much smaller than the number of inactives in the data set
* Precision - indicates the ratio of predicted actives that were actually actives
* Recall - indicates the ratio of the predicted actives per actual actives.


## Testing models

The procedure of testing is fundamental as it allows to evaluate how well the model responds to an external dataset where we know the actual activiy values. If the previous option allows us to do an actual N-fold cross validation, this option will allow us to validate it with an independent validation set

```sh 
$ python msqsar.py test -in ..\..\data\H1\H1_train.sar -test ..\..\data\H1\H1_test.sar
```

As it can be seen, we just need to specify the name of the test file with the `-test` option. This produces the following output

```
Testing!
Model Type: Regression
RMSE:  0.1418
Pearson:  0.8867
PVE:  0.7861
%Predicted:  0.862
```

The statistics are the same as above, but now refer to the validation set. As it can be easily verified, the results are highly similar, suggesting that the model is actually capable of making reliable predictions for this dataset. 


| Statistic | Train | Test  |
| --------- | ----- | ----  |
|RMSE       | 0.1456| 0.1418|
|Pearson    | 0.8926| 0.8867|
|PVE        | 0.7960| 0.7861|
|*%Predicted* | 0.8870| 0.8620|



The actual prediction results can be obtained by using the `-detail` option and writing the output to a file with the `-out` option

```sh 
$ python msqsar.py test -in data/H1_train.sar -test data/H1_test.sar -silent -detail -out H1.preds
```

The `H1.preds` file is a text file that will include the model statistics as well as the individual prediction results for each molecule. Its first lines are something like this

```
RMSE:  0.1418
Pearson:  0.8867
PVE:  0.7861
%Predicted:  0.862
Detailed results:
   CHEMBL270177	 0.2397	NA
   CHEMBL1794855	 0.0000	NA
   CHEMBL90063	 0.0000	0.0000
   CHEMBL250403	 0.1747	NA
   CHEMBL343324	 0.2750	NA
   CHEMBL609579	 0.1399	0.2964
   ...
```

In the section *Detailed results:* (not completely displayed above) the actual data from the output .sar file is presented along the predicted values in the following column. We can see that, for the first two molecules, the model was not able to make any type of prediction and therefore assigned them an `NA` 

## Screening a database

This is one of the most powerful features of `msqsar` and it allows screening a database with millions of compounds for activity for a given target. For the moment it allows only the identification of potentially active molecules, and not so much the attribution of a quantitative score to each one. This can be easily solved as we will see.

The screening process, depending on the size of the screening database can be a very intensive computational process and as such it has been optimized to take advantage of parallel processing. This allows screening of a dataset with ~13 million compounds in 4-5 hours, using only a common desktop computer

Screening a database requires that the database is in SMI format [SMILES format](https://open-babel.readthedocs.io/en/latest/FileFormats/SMILES_format.html) with the SMILES of molecule and the molecule ID in each line. E.g.

```
COc1noc2c1CCNCC2 CHEMBL2331792
Cn1oc2c(c1=O)CCNCC2 CHEMBL2331804	
NC1=Nc2ccc(Cl)cc2CN1 CHEMBL2436555
```

The screening process is a 5 phase procedure
1. Checks that the model is a classification model: That is we want for thsi proces only identify whether or not each molecule in a data set is active or not. We are not ( at this phase) trying to assess whether how much the molecule is actually active
2. Determinte the structures of an input file that will be the testbed for data screening. It's agains this data set that the data in the screening database will be tested
3. Compute the Chemical space of the input data set
4. Preescreen. At this phase all the molecules will be verified if they have any likelihood  of being active. This is the lengthiest phase and largelly may benefit  from parallel processing
5. Screening. With the candidates determined, we can actually screen them assessing the likelihood of them being active for the existing QAR problem. This will output a SAR file with the final results for the process.

All the above procedures are accomplished in one only call. Here we will screen the `mini_zinc.smi` database with about 64k molecules to see home many may be suitable for an EGFR inhibition dataset. This sample database was created with randomly selected data from the [ZINC20 database](https://zinc20.docking.org/)

```
$ python msqsar.py screen -scr data/mini_zinc.smi -in data/EGFR_class.sar
```
Ant this is the output

```
Screening!
1. Model Type: Classification
2. Determining structure 6255
3. Computing chemical space
4. Starting pre-screen
        (1) Mols read:     10000 - Candidates found:     1 ( 0.0001)
        (1) Mols read:     20000 - Candidates found:     2 ( 0.0001)
        (1) Mols read:     30000 - Candidates found:     3 ( 0.0001)
        (1) Mols read:     40000 - Candidates found:     4 ( 0.0001)
        (1) Mols read:     50000 - Candidates found:     5 ( 0.0001)
        (1) Mols read:     60000 - Candidates found:     9 ( 0.0001)
5. Pre-screen finished. Candidates found: 9
6. Start the candidate screening...
16691784        A       CCOc1ccc(Nc2nncc3ccccc23)cc1
11906948        N       COc1cc2ncnc(Nc3ccccc3N3CCOCC3)c2cc1OC
12748174        NA      C[C@H](Nc1ncnc2scc(-c3ccccc3)c12)c1ccc(S(C)(=O)=O)cc1
89423861        NA      COc1cc(F)cc(NC(=O)Nc2ccccn2)c1
44208           N       C[C@H](Oc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1)C(=O)O
31948967        NA      COc1cc(CNc2ncnc3ccccc23)cc(OC)c1OC
8764851         N       COC[C@@H](O)CNC(=O)COc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1
170623036       A       Cc1ccc2ncnc(N[C@H](C)c3ccccc3)c2c1
38245886        A       OCCOCCn1ccc2ncnc(Nc3ccc(Oc4cccc(C(F)(F)F)c4)c(Cl)c3)c21
```

The final molecules might be saved on a .SAR file using the `-out` control parameter

The same procedure could be processed in parallel using the `-parallel` option in the `-nprocs` control parameter. This will create some data partitions of the database, but the procedure will be much faster

```
$ python msqsar.py screen -scr data/mini_zinc.smi -in data/EGFR_class.sar -parallel  -nprocs 6
```

Producing the following output
```
Screening!
1. Model Type: Classification
2. Determining structure 6255
3. Computing chemical space
4. Starting pre-screen
        This time it is parallel!
        (3) Mols read:     10000 - Candidates found:     2 ( 0.0002)
        (1) Mols read:     10000 - Candidates found:     0 ( 0.0000)
        (2) Mols read:     10000 - Candidates found:     4 ( 0.0004)
        (6) Mols read:     10000 - Candidates found:     0 ( 0.0000)
        (4) Mols read:     10000 - Candidates found:     1 ( 0.0001)
        (5) Mols read:     10000 - Candidates found:     2 ( 0.0002)
5. Pre-screen finished. Candidates found: 9
6. Start the candidate screening...
16691784        A       CCOc1ccc(Nc2nncc3ccccc23)cc1
11906948        N       COc1cc2ncnc(Nc3ccccc3N3CCOCC3)c2cc1OC
31948967        NA      COc1cc(CNc2ncnc3ccccc23)cc(OC)c1OC
170623036       A       Cc1ccc2ncnc(N[C@H](C)c3ccccc3)c2c1
12748174        NA      C[C@H](Nc1ncnc2scc(-c3ccccc3)c12)c1ccc(S(C)(=O)=O)cc1
8764851         N       COC[C@@H](O)CNC(=O)COc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1
89423861        NA      COc1cc(F)cc(NC(=O)Nc2ccccn2)c1
44208           N       C[C@H](Oc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1)C(=O)O
38245886        A       OCCOCCn1ccc2ncnc(Nc3ccc(Oc4cccc(C(F)(F)F)c4)c(Cl)c3)c21
```


## Using `msqsar` as a library

The file `msqsar.py` can be used as a Python module provided it is accessible within the path or in the same folder of the application

### Example 1: Validating a model

For instance for validating a model against a testing set, with all default parameters but a max distance = 0.8, we could run
```python
import msqsar as msq
res = msq.validate_model("data/H1_train.sar", "data/H1_test.sar", max_dst=0.8)
stats=msq.get_stats(res)
for st in stats: print(st, "-->", stats[st])
```

which will produce:

```
Model Type: Regression
pve --> 0.7012722414696078
rmse --> 0.17155424829084787
pearson --> 0.8397271370003132
N --> 246
N0 --> 243
predicted --> 0.9878048780487805
```

### Example 2: Screening a database

For screening a database we call the `screener()` function. **IMPORTANT NOTE:** within Jupyter the parallel processing will probably not work

```python
import msqsar as msq
res = msq.screener("data/bEGFR_class.sar", "data/mini_zinc.smi", max_dst=0.5)

#sort of SAR file printing
for mid in res['preds']:
    print(mid, "\t", res['preds'][mid], "\t", res['smiles'][mid])
```

will give:

```
16691784 	 A 	 CCOc1ccc(Nc2nncc3ccccc23)cc1
11906948 	 A 	 COc1cc2ncnc(Nc3ccccc3N3CCOCC3)c2cc1OC
12748174 	 NA 	 C[C@H](Nc1ncnc2scc(-c3ccccc3)c12)c1ccc(S(C)(=O)=O)cc1
89423861 	 NA 	 COc1cc(F)cc(NC(=O)Nc2ccccn2)c1
44208 	        N 	 C[C@H](Oc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1)C(=O)O
31948967 	 A 	 COc1cc(CNc2ncnc3ccccc23)cc(OC)c1OC
8764851 	 N 	 COC[C@@H](O)CNC(=O)COc1cc(O)c2c(=O)cc(-c3ccccc3)oc2c1
170623036 	 A 	 Cc1ccc2ncnc(N[C@H](C)c3ccccc3)c2c1
38245886 	 A 	 OCCOCCn1ccc2ncnc(Nc3ccc(Oc4cccc(C(F)(F)F)c4)c(Cl)c3)c21```
```


