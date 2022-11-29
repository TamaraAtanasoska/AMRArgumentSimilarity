# Reproduction study: Argument Similarity with AMR

This project reproduces the paper "Explainable Unsupervised Argument Similarity Rating with Abstract Meaning Representation and Conclusion Generation"(Opitz et al., ArgMining 2021) [(link)](project_docs/OpitzEtAl21.pdf). 

To reproduce the results of the study, two repositories originally written by the author are required: [a more general repository](https://github.com/flipz357/amr-metric-suite) that contains AMR metrics and the [paper specific repository](https://github.com/Heidelberg-NLP/amr-argument-sim) that contains the AMR metric for argument similarity. As we focus on the results obtained in the paper, we provide both repositories cloned in [```repro_repos/```](repro_repos/). Additionally, we use the [AMR parser](https://github.com/bjascob/amrlib) as used in the paper as library to turn sentences into graphs.

It will also include an extension. Currently a work in progress as we gather ideas.

This project is a so called "Project Module", part of the [Cognitive Systems](https://www.uni-potsdam.de/en/studium/what-to-study/master/masters-courses-from-a-to-z/cognitive-systems) Master program at the [University of Potsdam](https://www.uni-potsdam.de/en/university-of-potsdam). Contributors: [Tamara Atanasoska](https://github.com/TamaraAtanasoska), [Emanuele De Rossi](https://github.com/EmanueleDeRossi1) and [Galina Ryazanskaya](https://github.com/flying-bear).

## Setup

### Environment recreation 

In the folder ```setup/``` you can find the respective environment replication and package requirements files. There are two options:

  1. You can run ```pip install -r setup/requirements.txt``` to install the necessary packages in your existing environment.

  2. If you are using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage your virtual environments, you can replicate and activate the full exact environment with the following commands:

   ```
   conda env create --name <name> --file setup/conda.yml
   conda activate <name>
   ```
   
## Generating the S2match scores and evaluating with the AMR metric

### Generating S2match scores

To get the results using the S2match metric, which is denoted as "standard" in the paper you can run: 
```
cd repro_repos/amr-metric-suite/py3-Smatch-and-S2match
python smatch/s2match.py -f <file1> <file2> -cutoff 0.95 --ms
```
The cutoff parameter's value 0.95 corresponds to the value used in the paper. A high cutoff parameter allows score to increase only for (near-)paraphrasal concepts.

You would need to save the output in a file to be able to use it to evaluate with the AMR metric below. You could either copy paste the output, or run the following complete command: 
```
cd repro_repos/amr-metric-suite/py3-Smatch-and-S2match
python smatch/s2match.py -f <file1> <file2> -cutoff 0.95 --ms > s2match_scores_standard.txt
```
### Evaluating with the AMR metric

In the [```sim_preds/```](repro_repos/amr-argument-sim/scripts/sim_preds) folder there are various predictions stored. To test-evaluate all of them with the AMR metric, you can run the command below. You will see the ouput as the command runs.  
```
cd repro_repos/amr-argument-sim/scripts/
./evaluate_all.sh
```
What the script does, is use all the files that are in the similarity predictions folder to evaluate. If you would like to use a file of your own, you can use the [```evaluate.py```](repro_repos/amr-argument-sim/scripts/evaluate.py) script. 

## Using the AMR parser

If you have set up your environment with [our section](#environment-recreation) about it above, you will already have all the packages installed. If you haven't and you would like to install only the relevant ones for this parser, please take a look at the parser's [installation guide](https://amrlib.readthedocs.io/en/latest/install/). If you only need the parser, it is a better idea to clone the code from the [original repository](https://github.com/bjascob/amrlib) too. 

After installing all the required packages by any of the means, run: 
```
pip install amrlib
```
This will install the library. However, in order to parse, you will also need to pick and download a model to do that with. All the models currently available are found in [this repository](https://github.com/bjascob/amrlib-models). 

We picked to use the [oldest T5 based parse model](https://github.com/bjascob/amrlib-models/releases/tag/model_parse_t5-v0_1_0) available, to be able to get as comparable results with the ones in the paper as possible. 

Once you have picked the model, you need to download it, extract it and rename it. More information [here](https://amrlib.readthedocs.io/en/latest/install/#install-the-models). On Windows it would be easier to just download the zip file, extract and rename the folder instead of the last linking command. 
```
pip show amrlib #copy path where the package is stored 
cd <path-to-where-the-package-is-stored>

mkdir data
cd data
tar xzf <model-filename> #copy file here before running command
ln -snf <model-filenam>  model_stog
```

To test if the parser is working and the installation is correct, you can run: 
```
cd scripts
python ./test_parse.py 
```
